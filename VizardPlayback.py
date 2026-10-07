"""Inspect/decimate Basilisk length-delimited protobuf playback without Vizard.

Unknown protobuf fields are retained byte-for-byte when making a smaller copy.
The reader checks framing and spacecraft state, not every Vizard schema field.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def varint(data, offset=0):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError('Truncated protobuf varint')
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError('Invalid protobuf varint')


def fields(data):
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        number, wire = key >> 3, key & 7
        if not number:
            raise ValueError('Invalid zero protobuf field')
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 2, 5):
            size = {1: 8, 5: 4}.get(wire)
            if wire == 2:
                size, offset = varint(data, offset)
            if offset + size > len(data):
                raise ValueError('Truncated protobuf field')
            value = data[offset:offset+size]
            offset += size
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        yield number, wire, value


def frames(path):
    with Path(path).open('rb') as stream:
        while True:
            prefix = stream.read(1)
            if not prefix:
                return
            while prefix[-1] & 128:
                byte = stream.read(1)
                if not byte or len(prefix) >= 5:
                    raise ValueError(f'Truncated/invalid frame prefix at byte {stream.tell()}')
                prefix += byte
            size, _ = varint(prefix)
            if size <= 0 or size > 64*1024*1024:
                raise ValueError(f'Invalid playback frame length {size}')
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f'Truncated playback frame at byte {stream.tell()}')
            yield prefix + payload, list(fields(payload))


def doubles(data):
    if len(data) % 8:
        raise ValueError('Invalid packed doubles')
    return struct.unpack('<' + 'd'*(len(data)//8), data)


def encode_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_fields(items):
    result = bytearray()
    for n,w,v in items:
        result.extend(encode_varint(n*8+w))
        if w == 0:
            result.extend(encode_varint(v))
        else:
            if w == 2:
                result.extend(encode_varint(len(v)))
            result.extend(v)
    return bytes(result)


def repair_buffered_playback(source, output):
    """Remove obsolete thrust overlay and request full-file playback buffering.

    All physical states, engine outputs, timestamps and ephemerides are retained.
    The original is never overwritten.
    """
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():
        raise ValueError('Output must differ from original playback')
    removed={b'Main thrust origin',b'Main thrust tip'}
    count=0
    with output.open('xb') as stream:
        for _,items in frames(source):
            result=[]
            for n,w,b in items:
                if n==3 and any(k==1 and v in removed for k,_,v in fields(b)):
                    continue
                if n==7:
                    settings=[]
                    for k,kw,v in fields(b):
                        if k==53:  # int64 messageBufferSize
                            continue
                        if k in (5,12) and any(j in (1,2) and value in removed for j,_,value in fields(v)):
                            continue
                        settings.append((k,kw,v))
                    settings.append((53,0,(1<<64)-1))  # signed -1: full-file buffer
                    b=encode_fields(settings)
                result.append((n,w,b))
            payload=encode_fields(result)
            stream.write(encode_varint(len(payload))+payload)
            count+=1
    return count


def playback_frame(items, ratings=None, minimal=False):
    """Change display metadata only; spacecraft states and timestamps stay intact."""
    adjusted = []
    for n,w,v in items:
        if n == 3 and ratings:
            sc = []
            for sn,sw,sv in fields(v):
                if sn == 6:
                    thr = list(fields(sv))
                    name = next((b.decode('utf-8') for k,_,b in thr if k == 8), '')
                    if name in ratings:
                        thr = [(k,kw,b) for k,kw,b in thr if k != 6]
                        thr.append((6,1,struct.pack('<d',ratings[name])))
                        sv = encode_fields(thr)
                sc.append((sn,sw,sv))
            v = encode_fields(sc)
        if n == 7 and minimal:
            v = encode_fields([(k,kw,b) for k,kw,b in fields(v) if k != 12])
        adjusted.append((n,w,v))
    payload = encode_fields(adjusted)
    return encode_varint(len(payload)) + payload


def state_from_frame(items, name='IMX'):
    timestamp = None
    state = None
    for number, wire, value in items:
        if number == 1:
            timestamp = next((doubles(v)[0]*1e-9 for n,w,v in fields(value) if n == 2), None)
        if number == 3:
            sc = list(fields(value))
            label = next((v.decode('utf-8') for n,w,v in sc if n == 1), '')
            if label == name:
                state = {n: doubles(v) for n,w,v in sc if n in (2,3,4) and w in (1,2)}
    if timestamp is None or state is None or any(n not in state for n in (2,3,4)):
        raise ValueError(f'Missing timestamp or spacecraft state for {name}')
    if not all(math.isfinite(x) for x in (timestamp, *state[2], *state[3], *state[4])):
        raise ValueError(f'Non-finite spacecraft state at {timestamp} s')
    if len(state[2]) != 3 or len(state[3]) != 3:
        raise ValueError('Invalid position/velocity dimensions')
    return timestamp, state


def inspect_playback(source, output=None, interval_s=.2, target=(0.,-1737401.,-50.), radius=1737400.,
                     ratings=None, minimal=False):
    source = Path(source)
    if output is not None and Path(output).resolve() == source.resolve():
        raise ValueError('Output must differ from original playback')
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError('Playback interval must be positive')
    output_stream = Path(output).open('xb') if output is not None else None
    count = saved = 0
    last_saved = -math.inf
    previous = -math.inf
    rows = []
    min_height = math.inf
    first_surface_crossing = None
    first_frame = None
    metadata = {}
    raw = None
    try:
        for raw, items in frames(source):
            t, state = state_from_frame(items)
            if t <= previous:
                raise ValueError(f'Non-increasing playback time: {t}')
            previous = t
            count += 1
            if first_frame is None:
                first_frame = items
            r, v = state[2], state[3]
            height = math.hypot(*r) - radius
            min_height = min(min_height, height)
            if height < 0 and first_surface_crossing is None:
                first_surface_crossing = t
            # Keep the first frame (settings/models), global configuration changes,
            # regular samples, and the last frame. Current project metadata is
            # initialized in the first frame; it does not switch spacecraft later.
            global_fields = {}
            for n,w,b in items:
                if n >= 4:
                    global_fields.setdefault(n, []).append((w,b))
            changed = any(metadata.get(n) != values for n,values in global_fields.items())
            metadata.update(global_fields)
            keep = t - last_saved >= interval_s - 1e-8 or changed
            if keep:
                if output_stream:
                    output_stream.write(playback_frame(items,ratings,minimal) if ratings or minimal else raw)
                saved += 1
                last_saved = t
                rows.append((t,height,math.hypot(*v),math.dist(r,target),*r,*v))
        if count < 2:
            raise ValueError('Playback needs at least two frames')
        if last_saved < t:
            if output_stream:
                output_stream.write(playback_frame(items,ratings,minimal) if ratings or minimal else raw)
            saved += 1
            rows.append((t,height,math.hypot(*v),math.dist(r,target),*r,*v))
    finally:
        if output_stream:
            output_stream.close()
    summary = dict(source=str(source.resolve()), bytes=source.stat().st_size, frames=count,
                   duration_s=t, sampled_frames=saved, min_reference_altitude_m=min_height,
                   first_below_reference_surface_s=first_surface_crossing,
                   final_reference_altitude_m=height, final_speed_m_s=math.hypot(*v),
                   final_target_error_m=math.dist(r,target), final_position_m=r, final_velocity_m_s=v,
                   structurally_complete=True, all_recorded_states_finite=True)
    if output is not None:
        summary.update(playback_copy=str(Path(output).resolve()), copy_bytes=Path(output).stat().st_size)
    return summary, rows, first_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output')
    parser.add_argument('--interval',type=float,default=.2)
    args = parser.parse_args()
    summary, _, _ = inspect_playback(args.source,args.output,args.interval)
    print(json.dumps(summary,indent=2))
