"""Full-episode POV masks seeded by Molmo and recovered from TAPIR tracks."""
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from robot101.data.utilities.annotation_tree import atomic_json
from robot101.perception.gripper_masks import BODY_PROMPT, NEGATIVE_PROMPT, valid_points, filter_candidates


def prepare_episode_masks(frames, real, molmo, tapir, sam, root, episode, corrections, args, *, fps=30.0):
    root = Path(root)
    directory, review = root / 'gripper_masks', root / 'mask_review'
    directory.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    real = np.asarray(real, bool)
    refresh_fps = float(args.sam2_gripper_refresh_fps)
    if refresh_fps > fps:
        raise ValueError('POV gripper refresh FPS cannot exceed source FPS')
    # Optional SAM re-anchors use tracked TAPIR points. Molmo is called only to
    # initialize the episode (or when the initial query itself is unavailable).
    refresh_frames = (np.unique(np.rint(np.arange(0, len(frames), fps / refresh_fps)).astype(int))
                      if refresh_fps > 0 else np.asarray([], int))
    refresh_frames = refresh_frames[refresh_frames < len(frames)].tolist()
    manual = {}
    for key, value in corrections.items():
        if key.split(':', 1)[0] == str(episode):
            frame = int(value['frame_index'])
            if not 0 <= frame < len(frames) or not real[frame]:
                raise ValueError(f'Human POV correction outside valid episode: {key}')
            if frame in manual and manual[frame] != value:
                raise ValueError(f'Conflicting human POV corrections on frame {frame}')
            manual[frame] = value
    prompts, track_banks = [], []

    def cache_tapir_track(seed_frame, points):
        seed_frame = int(seed_frame)
        points = np.asarray(points, np.float32).reshape(-1, 2)
        if not len(points):
            raise ValueError('no_positive_gripper_points_for_tapir')
        clip = frames[seed_frame:]
        tracks, visible, _, _ = tapir.track_video(
            clip, points, query_frame_index=0)
        track_banks.append((seed_frame, points.copy(), tracks, visible))
        return tracks, visible

    def tracked_points(frame):
        for seed_frame, _, tracks, visible in reversed(track_banks):
            local = int(frame) - seed_frame
            if not 0 <= local < tracks.shape[1]:
                continue
            xy = tracks[:, local]
            keep = visible[:, local] & np.isfinite(xy).all(axis=1)
            keep &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            if keep.any():
                return xy[keep].astype(np.float32)
        return np.empty((0, 2), np.float32)

    # Find one initial positive prompt set, then let TAPIR carry those points
    # through the episode. No periodic Molmo calls are scheduled.
    initial_seed = None
    initial_points = np.empty((0, 2), np.float32)
    for frame in np.flatnonzero(real)[:args.sam2_seed_search_frames]:
        frame = int(frame)
        if frame in manual:
            value = manual[frame]
            points = valid_points({'points_xy': value['points_xy']}, w, h)
            if len(points) == len(value['points_xy']) and len(points):
                initial_seed, initial_points = frame, points
                break
        print(f'  POV Molmo ep{episode}: initial gripper detection at frame {frame}', flush=True)
        result = molmo.ground(frames[frame], BODY_PROMPT)
        points = valid_points(result, w, h)
        prompts.append(dict(frame=frame, role='gripper_body_initial', points_xy=points.tolist(), error=result.get('error')))
        if len(points):
            initial_seed, initial_points = frame, points
            break
    if initial_seed is not None:
        cache_tapir_track(initial_seed, initial_points)

    def initialize(cursor):
        if cursor in manual:
            value = manual[cursor]
            points = valid_points({'points_xy': value['points_xy']}, w, h)
            if len(points) != len(value['points_xy']):
                raise ValueError('invalid_human_coordinates')
            labels = np.asarray(value.get('labels', [1]*len(points)), np.int32)
            if labels.shape != (len(points),) or not (labels == 1).any():
                raise ValueError('human_correction_requires_positive_gripper_points')
            cache_tapir_track(cursor, points[labels == 1])
            return cursor, points, labels.tolist(), True
        if initial_seed is not None and cursor <= initial_seed:
            return initial_seed, initial_points, [1]*len(initial_points), False
        points = tracked_points(cursor)
        if len(points):
            return int(cursor), points, [1]*len(points), False
        raise ValueError('tapir_gripper_track_unavailable')

    def refine(frame, provisional, points, labels):
        if not args.sam2_negative_prompts or frame != initial_seed:
            return points, labels
        image = frames[frame].copy()
        image[provisional] = (0.65*image[provisional] + 0.35*np.array([255, 0, 255])).astype(np.uint8)
        for xy in points:
            cv2.circle(image, tuple(np.rint(xy).astype(int)), 4, (0, 255, 0), -1)
        cv2.imwrite(str(directory / f'ep{episode:06d}_seed{frame:06d}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        raw = molmo.ground(image, NEGATIVE_PROMPT)
        negatives = valid_points(raw, w, h)
        prompts.append(dict(frame=frame, role='object_negatives', points_xy=negatives.tolist(), error=raw.get('error')))
        if raw.get('error'):
            raise ValueError(f"negative_prompt_failed:{raw['error']}")
        if not len(negatives) and args.sam2_require_negative_points:
            raise ValueError('object_negative_points_unavailable')
        return np.concatenate([points, negatives]), np.asarray([*labels, *([0]*len(negatives))], np.int32)

    def validate(frame, mask, seed_area, previous_area, points, labels):
        if mask.shape != (h, w) or mask.sum() < 8:
            return 'empty_or_invalid_gripper_mask'
        area = float(mask.mean())
        if area > args.sam2_max_area_fraction:
            return 'excessive_gripper_mask'
        for baseline in (seed_area, previous_area):
            if (baseline is not None and area > baseline * args.sam2_max_growth_ratio
                    and area - baseline > args.sam2_growth_area_fraction):
                return 'suspicious_gripper_mask_growth'
        if points is not None:
            pixels = np.rint(points).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, w-1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, h-1)
            inside = mask[pixels[:, 1], pixels[:, 0]]
            if not inside[np.asarray(labels) == 1].all():
                return 'mask_misses_positive_prompt'
        return None

    started = perf_counter()
    print(f'  SAM 2 ep{episode}: full POV episode, {len(frames)} frames', flush=True)
    if not real.any():
        masks = np.zeros((len(frames), h, w), bool)
        valid = np.zeros(len(frames), bool)
        events = [dict(frame=0, status='needs_review', reason='no_valid_wrist_frames')]
    else:
        try:
            masks, valid, events = sam.propagate_episode(
                frames, real, initialize, refine, validate,
                max_recoveries=args.sam2_max_recoveries, retry_frames=args.sam2_recovery_retry_frames,
                correction_frames=manual, refresh_frames=refresh_frames)
        except Exception as exc:
            masks = np.zeros((len(frames), h, w), bool)
            valid = np.zeros(len(frames), bool)
            events = [dict(frame=0, status='needs_review', reason=f'episode_sam_failed:{exc}')]
    path = directory / f'ep{episode:06d}_full.npz'
    np.savez_compressed(path, packed_masks=np.packbits(masks, axis=-1), shape=np.asarray(masks.shape),
                        frame_indices=np.arange(len(frames)), valid_frames=valid)
    metadata = dict(status='ok' if valid.any() else 'failed', scope='full_episode_pov',
        camera='wrist', episode=episode, window=[0, len(frames)-1],
        mask_file=path.relative_to(root).as_posix(), model=sam.model_id,
        mask_seconds=perf_counter()-started, valid_frame_count=int(valid.sum()),
        tapir_reanchor_fps=refresh_fps, reanchor_source='tapir_track',
        initial_localization_source='molmo_or_human_correction',
        tapir_track_seeds=[dict(frame=int(seed), point_count=int(len(points)))
                           for seed, points, _, _ in track_banks],
        scheduled_tapir_reanchor_frames=refresh_frames,
        episode_manual_review_required=not bool(valid.all()),
        events=events, prompts=prompts)
    for event in events:
        if event['status'] != 'needs_review':
            continue
        frame = min(len(frames)-1, max(0, int(event['frame'])))
        if valid[frame]:
            # Successful same-frame recovery remains in the audit log, not the correction queue.
            continue
        stem = f'ep{episode:06d}_stage_recovery{frame:06d}'
        image_name = f'{stem}.png' if real[frame] else ''
        if image_name:
            cv2.imwrite(str(review / image_name), cv2.cvtColor(frames[frame], cv2.COLOR_RGB2BGR))
        atomic_json(review / f'{stem}.json', dict(camera='wrist', episode=episode, frame_index=frame,
            correction_key=f'{episode}:frame:{frame}', window=[0, len(frames)-1],
            review_image=image_name, reason=event['reason'], prompt=BODY_PROMPT,
            manual_review_required=True))
    atomic_json(directory / f'ep{episode:06d}_full.json', metadata)
    print(f'  SAM 2 ep{episode}: {int(valid.sum())}/{len(frames)} valid POV masks; '
          f'{sum(e["status"] == "needs_review" for e in events)} review events', flush=True)
    return masks, valid, metadata


def apply_episode_masks(entry, masks, valid, metadata, max_weight, radius_fraction, tail_frames):
    start, length = int(entry['tail_start']), entry['tracks'].shape[1]
    window_masks, valid_window = masks[start:start+length], valid[start:start+length]
    info = {**metadata, 'clustering_window': [start, start+length-1],
            'max_weight': max_weight, 'radius_fraction': radius_fraction,
            'manual_review_required': not bool(valid_window.all())}
    if not valid_window.all():
        entry['visible'][:] = False
        entry['eligible'] = np.zeros(len(entry['tracks']), bool)
        info['reason'] = 'unresolved_episode_mask_in_clustering_window'
    elif len(entry['tracks']):
        tracks, visible, weights, eligible, endpoints = filter_candidates(
            entry['tracks'], entry['visible'], window_masks, max_weight, radius_fraction, tail_frames)
        entry.update(tracks=tracks, visible=visible, observation_weights=weights, eligible=eligible, endpoints=endpoints)
        info['retained_candidates'] = int(eligible.sum())
    entry['gripper_filter'] = info
