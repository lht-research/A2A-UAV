import collections
import concurrent.futures
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import sys

QUICK = False

SPLIT_COUNTS = {'train': 61767, 'val': 16468, 'test': 27059}
PREFIXES = {'train': 'tr', 'val': 'va', 'test': 'te'}
META_FIELDS = {'sample_id', 'split', 'classification_label', 'spectrum_readable', 'recording_key', 'canonical_recording_key', 'family_key', 'session', 'scene', 'condition', 'timestamp', 'annotations', 'image_size', 'modalities', 'radar_pointcloud'}
MANIFEST_FIELDS = {'sample_id', 'split', 'sample_dir', 'metadata', 'files', 'recording_key', 'canonical_recording_key', 'family_key', 'session', 'scene', 'condition', 'timestamp', 'annotations', 'classification_label', 'spectrum_readable', 'radar_frames'}
INDEX_ROW_FIELDS = {'sample_id', 'split', 'path', 'label', 'canonical_recording_key', 'spectrum_readable'}
COCO_IMAGE_FIELDS = {'id', 'file_name', 'width', 'height', 'sample_id', 'split', 'modality'}


def verify_sample(item):
    root, split, row = item
    sid = row['sample_id']; rel = f'images/{split}/{sid}'
    assert re.fullmatch(PREFIXES[split] + r'_\d{6}', sid), (split, sid)
    assert set(row) == MANIFEST_FIELDS and row['split'] == split and row['timestamp'] is None
    assert row['sample_dir'] == rel and row['metadata'] == rel + '/metadata.json'
    sample = root / rel
    for name, path in row['files'].items():
        assert path == rel + '/' + name and not Path(path).is_absolute() and '..' not in Path(path).parts
        assert (root / path).is_file(), path
    assert {p.relative_to(sample).as_posix() for p in sample.rglob('*') if p.is_file()} == set(row['files']), sid
    meta = json.loads((root / row['metadata']).read_text())
    assert set(meta) == META_FIELDS
    for key in ['sample_id', 'split', 'timestamp', 'annotations', 'canonical_recording_key', 'recording_key', 'family_key', 'session', 'scene', 'condition', 'classification_label', 'spectrum_readable']:
        assert meta[key] == row[key], (sid, key)
    for name in meta['modalities'].values():
        if name is not None: assert isinstance(name, str) and name in row['files']
    group = meta['radar_pointcloud']; frames = group['frames']
    assert group['window_ns'] == 120000000 and group['frame_count'] == len(frames) == len(row['radar_frames'])
    assert group['total_point_count'] == sum(f['point_count'] for f in frames)
    assert group['nonempty_frame_count'] == sum(f['point_count'] > 0 for f in frames)
    assert group['dt_min_ns'] == min((abs(f['dt_ns']) for f in frames), default=None)
    assert group['dt_max_ns'] == max((abs(f['dt_ns']) for f in frames), default=None)
    assert [f['dt_ns'] for f in frames] == sorted(f['dt_ns'] for f in frames)
    expected_paths = set(); frame_ids = []; header_only = 0
    for frame, declared in zip(frames, row['radar_frames']):
        frame_ids.append(frame['frame_id'])
        assert isinstance(frame['dt_ns'], int) and abs(frame['dt_ns']) <= 120000000
        assert declared == {**frame, 'file': rel + '/' + frame['file']}
        assert row['files'][frame['file']] == declared['file']
        expected_paths.add(frame['file'])
        if not QUICK:
            body = (root / declared['file']).read_bytes()
            assert hashlib.sha256(body).hexdigest() == frame['sha256'], (sid, frame['file'])
            data = list(csv.reader(io.StringIO(body.decode('utf-8'))))
            assert data[0] == ['x', 'y', 'z', 'intensity', 'velocity']
            assert sum(bool(r) for r in data[1:]) == frame['point_count']
        header_only += frame['point_count'] == 0
    assert expected_paths == {name for name in row['files'] if name.endswith('.csv')}
    if frames:
        nearest = min(frames, key=lambda f: (abs(f['dt_ns']), f['dt_ns']))
        assert nearest['file'].startswith('radar/') and nearest['file'].endswith('.csv')
        assert group['csv'] == meta['modalities']['radar_pointcloud_csv'] == nearest['file']
        assert group['point_count'] == nearest['point_count']
    else:
        assert group['csv'] is None and group['point_count'] == 0
        assert meta['modalities']['radar_pointcloud_csv'] is None
    assert 'radar_pointcloud.csv' not in row['files'] and not (sample / 'radar_pointcloud.csv').exists()
    for ann in row['annotations'].values():
        assert len(ann['boxes']) == len(ann['labels']) <= 1, (sid, 'multiple boxes prohibited in the release')
        assert all(label == row['classification_label'] for label in ann['labels'])
    assert not re.search(r'/(?:root|mnt|tmp|home|Users)/|\.iq\.wav', json.dumps([row, meta], ensure_ascii=False))
    return sid, len(row['files']), frame_ids, header_only, not row['spectrum_readable']


def main(root):
    root = Path(root)
    rows = {}; ids = set(); all_frames = set(); groups = {}; unreadable = []; total_files = 0; radar_hist = collections.Counter(); header_only = 0; total_boxes = 0; empty_images = 0
    for split, count in SPLIT_COUNTS.items():
        source = [json.loads(line) for line in (root/'manifests'/f'full_{split}.jsonl').open() if line.strip()]
        assert len(source) == count
        rows[split] = {r['sample_id']: r for r in source}
        assert len(rows[split]) == count
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for n, result in enumerate(pool.map(verify_sample, ((root, split, r) for r in source)), 1):
                sid, file_count, frames, empty, bad_spectrum = result
                assert sid not in ids; ids.add(sid)
                for frame_id in frames:
                    assert frame_id not in all_frames; all_frames.add(frame_id)
                total_files += file_count; radar_hist[len(frames)] += 1; header_only += empty
                if bad_spectrum: unreadable.append(sid)
                if n % 5000 == 0: print('SAMPLES_VERIFIED', split, n, '/', count, flush=True)
        assert sum(p.is_dir() for p in (root/'images'/split).iterdir()) == count
        groups[split] = {k: {r[k] for r in source} for k in ['canonical_recording_key', 'recording_key', 'family_key']}
        print('MANIFEST_AND_FILES_OK', split, count, flush=True)
        coco = json.loads((root/'annotations'/f'instances_{split}.json').read_text())
        assert set(coco) == {'info', 'licenses', 'categories', 'images', 'annotations'}
        assert set(coco['info']) == {'description'} and len(coco['images']) == count * 2
        assert [c['id'] for c in coco['categories']] == list(range(1, 16))
        cat_names = {c['id']: c['name'] for c in coco['categories']}
        imap = {}; modality_counts = collections.Counter()
        for im in coco['images']:
            assert set(im) == COCO_IMAGE_FIELDS and im['id'] not in imap
            imap[im['id']] = im
            row = rows[split][im['sample_id']]
            assert im['modality'] in ['rgb', 'panorama']
            filename = 'rgb.png' if im['modality'] == 'rgb' else 'panorama_lr.png'
            assert im['file_name'] == row['files'][filename] and im['split'] == split
            modality_counts[im['modality']] += 1
        assert modality_counts == {'rgb': count, 'panorama': count}
        annotation_ids = set(); box_counts = collections.Counter(); geometries = collections.defaultdict(list)
        for ann in coco['annotations']:
            assert ann['id'] not in annotation_ids; annotation_ids.add(ann['id'])
            image = imap[ann['image_id']]; row = rows[split][image['sample_id']]
            x,y,w,h = ann['bbox']; assert h > 0 and 0.2 <= w/h <= 20; assert all(math.isfinite(v) for v in [x,y,w,h]) and w >= 1 and h >= 1 and w*h >= 4 and abs(w*h-ann['area']) < 1e-5
            assert x >= 0 and y >= 0 and x+w <= image['width']+1e-6 and y+h <= image['height']+1e-6
            assert cat_names[ann['category_id']] == row['classification_label']
            key = (image['sample_id'], image['modality']); box_counts[key] += 1
            geometries[key].append([x,y,x+w,y+h])
        for sid,row in rows[split].items():
            assert any(a['boxes'] for a in row['annotations'].values()), (sid, 'both modalities unannotated')
            for mod in ['rgb','panorama']:
                assert box_counts[(sid,mod)] == len(row['annotations'][mod]['boxes']) <= 1
                for a,b in zip(geometries[(sid,mod)],row['annotations'][mod]['boxes']):assert all(abs(x-y)<1e-4 for x,y in zip(a,b))
                empty_images += box_counts[(sid,mod)] == 0
        total_boxes += len(coco['annotations'])
        print('COCO_OK',split,len(coco['images']),len(coco['annotations']),flush=True)
    for a,b in itertools.combinations(SPLIT_COUNTS,2):
        for key in groups[a]:assert not groups[a][key] & groups[b][key],(a,b,key)
    assert len(ids) == sum(SPLIT_COUNTS.values()) and len(unreadable) == 0
    for name in ['classification_index.json','classification_index_trainable.json']:
        index=json.loads((root/'metadata'/name).read_text())
        assert set(index)=={'class_to_idx','counts','rows'} and len(index['class_to_idx'])==15
        for split in SPLIT_COUNTS:
            expected=set(rows[split])-(set(unreadable) if 'trainable' in name else set())
            assert index['counts'][split]==len(index['rows'][split])==len(expected)
            assert {r['sample_id'] for r in index['rows'][split]}==expected
            for row in index['rows'][split]:
                assert set(row)==INDEX_ROW_FIELDS
                source=rows[split][row['sample_id']]
                assert row['label']==source['classification_label'] and row['path']==source['files']['spectrum_tf.png']
                assert row['spectrum_readable']==source['spectrum_readable'] and row['canonical_recording_key']==source['canonical_recording_key']
    with (root/'metadata/recording_assignments.csv').open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['split','canonical_recording_key','classification_label','sample_count']; assignments=list(reader)
    assert len(assignments)==109
    actual=collections.Counter((sp,r['canonical_recording_key'],r['classification_label']) for sp in rows for r in rows[sp].values())
    assert actual=={(a['split'],a['canonical_recording_key'],a['classification_label']):int(a['sample_count']) for a in assignments}
    removal=json.loads((root/'reports/integrity_report.json').read_text())['quality_control']['excluded_multibox_samples']
    assert removal['removed_samples']==len(removal['records'])==33
    for r in removal['records']:
        assert r['sample_id'] not in ids and not (root/'images'/r['split']/r['sample_id']).exists()
    assert total_boxes==209539 and empty_images==1049 and len(all_frames)==505673
    cleanup=json.loads((root/'reports/integrity_report.json').read_text())['quality_control']['excluded_problem_samples']
    assert cleanup['removed_samples']==len(cleanup['records'])
    for r in cleanup['records']:
        assert r['sample_id'] not in ids and not (root/'images'/r['split']/r['sample_id']).exists()
    assert (root/'CITATION.cff').is_file() and not (root/'CITATION.md').exists()
    result={'status':'ok','sample_count':len(ids),'sample_counts':SPLIT_COUNTS,'image_records':len(ids)*2,
            'bounding_boxes':total_boxes,'empty_image_records':empty_images,'multibox_image_records':0,
            'declared_sample_files_checked':total_files,'unreadable_spectrum_sample_ids':unreadable,
            'detection_vs_classification_label_disagreements':{},'radar_frame_count_histogram':dict(sorted(radar_hist.items())),
            'unique_radar_frames':len(all_frames),'header_only_radar_frames':header_only,
            'checks':['All declared sample files exist; file inventories agree.', 'Manifest, metadata, COCO and classification indexes agree.',
                      'All radar SHA256 values, point counts, time gaps and unique IDs verified.', 'Recording, family and canonical recording groups are isolated.',
                      'All 576 excluded samples are absent; every retained sample has at least one RGB/panorama box, and each image record has at most one box.', 'COCO box geometry agrees with sample metadata.'],
            'limits':'All radar payloads are hashed; image payloads are not all decoded or rehashed.'}
    if QUICK:
        result['checks']=[x for x in result['checks'] if not x.startswith('All radar SHA256')]
    result['limits']='This script does not decode PNG pixels. Default mode hashes radar payloads; --quick skips payload reads; --checksums rehashes every packaged file.'
    return result, rows


ROOT_FILES={'README.md','README_zh.md','DATA_CARD.md','LICENSE.md','CITATION.cff','SHA256SUMS'}
ROOT_DIRS={'annotations','images','manifests','metadata','reports','scripts'}

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_public_layout(root,rows,all_payloads=False):
    root=Path(root)
    assert {p.name for p in root.iterdir()}==ROOT_FILES|ROOT_DIRS,'Unexpected/missing dataset-root entries'
    assert all((root/n).is_file() for n in ROOT_FILES)
    assert all((root/n).is_dir() for n in ROOT_DIRS)
    branch_files={
        'annotations':{'instances_train.json','instances_val.json','instances_test.json'},
        'manifests':{'full_train.jsonl','full_val.jsonl','full_test.jsonl','release_manifest.jsonl'},
        'metadata':{'classification_index.json','classification_index_trainable.json','recording_assignments.csv'},
        'reports':{'build_summary.md','integrity_report.json'},
        'scripts':{'verify_release.py'},
    }
    for directory,names in branch_files.items():
        assert {p.name for p in (root/directory).iterdir()}==names,(directory,'Unexpected/missing entries')
    seen=set()
    for line in (root/'manifests/release_manifest.jsonl').open():
        if not line.strip():continue
        row=json.loads(line);sp=row['split'];sid=row['sample_id']
        assert sid not in seen and row==rows[sp][sid],('release_manifest mismatch',sid)
        seen.add(sid)
    assert len(seen)==sum(SPLIT_COUNTS.values())
    expected={p for sp in rows for r in rows[sp].values() for p in r['files'].values()}
    extra=(ROOT_FILES-{'SHA256SUMS'})|{directory+'/'+name for directory,names in branch_files.items() for name in names}
    expected.update(extra)
    checksums={}
    with (root/'SHA256SUMS').open() as f:
        for line in f:
            digest,rel=line.rstrip('\n').split(None,1);rel=rel.strip().lstrip('*')
            assert re.fullmatch('[0-9a-f]{64}',digest)
            assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
            assert rel not in checksums,('Repeated checksum path',rel)
            checksums[rel]=digest
    assert set(checksums)==expected,('Checksum coverage mismatch',len(checksums),len(expected))
    for rel in extra:assert file_sha(root/rel)==checksums[rel],('Metadata checksum mismatch',rel)
    for sp in rows:
        for row in rows[sp].values():
            for frame in row['radar_frames']:assert frame['sha256']==checksums[frame['file']]
    checked_payloads=0
    if all_payloads:
        for rel,digest in checksums.items():
            if rel in extra:continue
            assert file_sha(root/rel)==digest,('Payload checksum mismatch',rel)
            checked_payloads+=1
            if checked_payloads%25000==0:print('PAYLOAD_HASHES_VERIFIED',checked_payloads,flush=True)
    return {'root_files':sorted(ROOT_FILES),'root_directories':sorted(ROOT_DIRS),'release_manifest_samples':len(seen),
            'checksum_entries':len(checksums),'metadata_files_hashed':len(extra),'all_payload_hashes_checked':bool(all_payloads),
            'payload_files_hashed_this_run':checked_payloads,'SHA256SUMS_self_excluded':True}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Verify the public dataset. Default checks metadata and every radar payload.')
    parser.add_argument('root',help='Extracted UAV_full_dataset_release directory')
    parser.add_argument('--quick',action='store_true',help='Check structure, sample metadata and checksum inventory; skip radar payload reads.')
    parser.add_argument('--checksums',action='store_true',help='Rehash every published sample payload, including all PNGs. This reads the full dataset.')
    args=parser.parse_args();QUICK=args.quick
    result,rows=main(args.root)
    result['public_layout']=verify_public_layout(args.root,rows,all_payloads=args.checksums)
    result['quick_mode']=QUICK
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
