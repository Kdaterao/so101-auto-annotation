"""The combined player shows one final annotation while keeping both cameras in sync."""
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from view_combined_dataset import episode_player, final_annotation_text
from robot101.data.utilities.debug_overlays import DebugOverlayRenderer


class CombinedViewerTests(unittest.TestCase):
    def player_html(self):
        records = [dict(start_frame=0, end_frame=30, brief_caption='The robot approaches the block.',
                        detailed_caption='Longer final caption.', caption='RAW MODEL CAPTION',
                        review_reasons=['INTERNAL REVIEW DETAILS']),
                   dict(start_frame=30, end_frame=60, brief_caption='The robot releases the part.')]
        with patch('streamlit.components.v1.html') as render:
            episode_player([('top', b'video1'), ('wrist', b'video2')], records, 30, 0)
        return render.call_args.args[0]

    def test_only_one_final_caption_is_used_without_raw_caption_fallback(self):
        self.assertEqual(final_annotation_text(dict(brief_caption=' Final sentence. ',
                                                   brief_action='Action', detailed_caption='Long text')),
                         'Final sentence.')
        self.assertEqual(final_annotation_text(dict(brief_action='Move toward the block')),
                         'Move toward the block')
        self.assertEqual(final_annotation_text(dict(detailed_caption='Legacy final caption.')),
                         'Legacy final caption.')
        self.assertEqual(final_annotation_text(dict(caption='Unaggregated raw caption')),
                         'Final annotation unavailable')

    def test_single_annotation_box_and_no_internal_caption_or_review_details(self):
        html = self.player_html()
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__(); self.annotations = 0
            def handle_starttag(self, tag, attrs):
                self.annotations += tag == 'div' and dict(attrs).get('id') == 'annotation'
        elements = Elements()
        elements.feed(html)
        self.assertEqual(elements.annotations, 1)
        self.assertIn('The robot approaches the block.', html)
        self.assertNotIn('Longer final caption.', html)
        self.assertNotIn('RAW MODEL CAPTION', html)
        self.assertNotIn('INTERNAL REVIEW DETAILS', html)
        self.assertNotIn('captions.forEach', html)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise the browser player')
    def test_waits_for_both_cameras_and_updates_final_caption_at_boundaries(self):
        script = re.search(r'<script>(.*?)</script>', self.player_html(), re.S).group(1)
        harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.listeners={};this.children=[];this.textContent='';this._time=0;
 this.readyState=0;this.seeking=false;this.paused=true;this.duration=2;this.ended=false;this.playCalls=0;}
 append(...items){this.children.push(...items);}
 setAttribute(){} load(){}
 addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
 removeEventListener(name,fn){this.listeners[name]=(this.listeners[name]||[]).filter(f=>f!==fn);}
 emit(name){for(const fn of [...this.listeners[name]||[]])fn();}
 get currentTime(){return this._time;}
 set currentTime(value){this._time=value;this.emit('seeked');}
 pause(){this.paused=true;}
 async play(){this.playCalls++;this.paused=false;}
}
const elements=Object.fromEntries(['grid','play','seek','status','time','annotation'].map(k=>[k,new Element(k)]));
global.document={getElementById:k=>elements[k],createElement:tag=>new Element(tag)};
global.requestAnimationFrame=()=>{};
'''
        checks = r'''
(async()=>{
 const settle=()=>new Promise(resolve=>setImmediate(resolve));
 assert.equal(videos.length,2);
 videos[0].readyState=4;videos[0].emit('canplaythrough');await settle();
 assert.equal(videos[0].playCalls,0);assert.equal(videos[1].playCalls,0);
 videos[1].readyState=4;videos[1].emit('canplaythrough');await settle();
 assert.equal(elements.annotation.textContent,'The robot approaches the block.');
 assert.ok(videos.every(v=>v.playCalls>0));
 elements.seek.value='30';elements.seek.oninput();elements.seek.onchange();await settle();
 assert.equal(videos[0].currentTime,1);assert.equal(videos[1].currentTime,1);
 assert.equal(elements.annotation.textContent,'The robot releases the part.');
 elements.play.onclick();assert.ok(videos.every(v=>v.paused));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run(['node'], input=harness + script + checks, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hiding_preview_labels_keeps_grounding_dots_and_clean_source_pixels(self):
        record = dict(subtask=0, start_frame=0, end_frame=1,
                      pov=dict(tracks=[[[40, 50], [40, 50]]], visibility=[[True, True]]),
                      third_person=dict(top=dict(source='semantic_object',
                          tracks=[[[40, 50], [40, 50]]], visibility=[[True, True]])))
        with TemporaryDirectory() as directory:
            renderer = DebugOverlayRenderer([record], Path(directory), show_masks=False, show_labels=False)
            clean = np.zeros((80, 100, 3), np.uint8)
            for camera in ('top', 'wrist'):
                image = renderer.render(clean, 0, camera)
                self.assertFalse(image[:30].any())
                np.testing.assert_array_equal(image[50, 40], [0, 255, 0])
            self.assertFalse(clean.any())


if __name__ == '__main__':
    unittest.main()
