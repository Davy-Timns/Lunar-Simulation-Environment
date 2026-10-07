"""Add missing Earth/Sun display ephemerides to a Moon-only recording.

Use the simulation's actual SPICE start epoch, which may differ from old
recordings' default Vizard epoch. Spacecraft and Moon messages are unchanged.
"""
import argparse
from pathlib import Path
import struct
import numpy as np
from Basilisk.utilities import simIncludeGravBody
from VizardPlayback import frames, fields, doubles, encode_fields, playback_frame


def body_names(items):
    return [next(b.decode() for k, _, b in fields(v) if k == 1).lower()
            for n, _, v in items if n == 2]


def validate_lunar_scene(items):
    names = body_names(items)
    if 'moon' in names and 'earth' not in names:
        raise ValueError("Moon is missing its Earth parent; Vizard cannot initialize its orbit")


def repair(source, destination, epoch):
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError('Do not overwrite the source recording')
    factory = simIncludeGravBody.gravBodyFactory()
    factory.createBodies(['sun', 'earth', 'moon'])
    spice = factory.createSpiceInterface(time=epoch, epochInMsg=True)
    if not spice.SPICELoaded:
        raise RuntimeError('Required SPICE kernels could not be loaded')
    spice.zeroBase = 'moon'
    spice.Reset(0)
    e = factory.epochMsg.read()
    epoch_bytes = encode_fields([(1,0,e.year), (2,0,e.month), (3,0,e.day),
                                (4,0,e.hours), (5,0,e.minutes),
                                (6,1,struct.pack('<d',e.seconds))])
    count = 0
    with Path(destination).open('xb') as output:
        for _, items in frames(source):
            stamp = next(v for n,_,v in items if n == 1)
            now = round(next(doubles(v)[0] for n,_,v in fields(stamp) if n == 2))
            spice.UpdateState(now)
            # Verify the supplied epoch/frame against the existing Moon attitude
            # before creating an inconsistent mixture of ephemerides.
            if count == 0:
                moon = next(v for n,_,v in items if n == 2 and
                            next(b for k,_,b in fields(v) if k == 1).lower() == b'moon')
                rotation = next(doubles(v) for n,_,v in fields(moon) if n == 4)
                actual = np.asarray(spice.planetStateOutMsgs[2].read().J20002Pfix).ravel()
                if not np.allclose(rotation, actual, atol=1e-9, rtol=0):
                    raise ValueError('SPICE epoch/frame does not match recorded Moon attitude')
            names = body_names(items)
            for index, name in enumerate(('sun', 'earth')):
                if name in names:
                    continue
                body = factory.gravBodies[name]
                state = spice.planetStateOutMsgs[index].read()
                payload = [(1,2,name.encode()),
                           (2,2,struct.pack('<3d',*state.PositionVector)),
                           (3,2,struct.pack('<3d',*state.VelocityVector)),
                           (4,2,struct.pack('<9d',*np.asarray(state.J20002Pfix).ravel())),
                           (5,1,struct.pack('<d',body.mu/1e9)),
                           (6,1,struct.pack('<d',body.radEquator/1000)),
                           (7,1,struct.pack('<d',body.radiusRatio))]
                items.append((2,2,encode_fields(payload)))
            if count == 0:
                items = [(n,w,v) for n,w,v in items if n != 8] + [(8,2,epoch_bytes)]
            validate_lunar_scene(items)
            output.write(playback_frame(items))
            count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    print(f'Repaired {repair(args.source, args.destination, args.epoch)} frames')
