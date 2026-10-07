"""Preloaded terrain geometry, independent of spacecraft truth or navigation.

Map coordinates are local X/Y horizontal, Z up, with origin at the south-pole
reference surface. Unmapped locations return None, never a fabricated height.
"""
from pathlib import Path
import hashlib
import numpy as np
import mujoco
from Spacecraft.Mujoco.FrameTransforms import R_BSK_TO_MUJOCO, R_MUJOCO_TO_BSK, TERRAIN_POSITION_BSK

MAP_TO_N = np.array([[1.,0.,0.], [0.,0.,-1.], [0.,1.,0.]])
BSK_TO_UNITY = np.array([[0.,1.,0.], [0.,0.,1.], [-1.,0.,0.]])


class TerrainMap:
    up_N = MAP_TO_N[:, 2]
    def __init__(self, altimeter):
        self.model = altimeter.model
        # Separate data prevents map queries from changing sensor/contact state.
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.geom = altimeter.terrain_geom_id
        self.body_exclude = altimeter.spacecraft_body_id
        if self.geom < 0:
            raise ValueError('Terrain map requires the lunar terrain mesh')
        mesh = self.model.geom_dataid[self.geom]
        start = self.model.mesh_vertadr[mesh]
        n = self.model.mesh_vertnum[mesh]
        vertices = np.array(self.model.mesh_vert[start:start+n],dtype=float)
        self.vertices_N = (vertices @ self.data.geom_xmat[self.geom].reshape(3,3).T
                           + self.data.geom_xpos[self.geom]) @ R_MUJOCO_TO_BSK.T
        vertices_M = self.to_map(self.vertices_N)
        self.minimum = vertices_M.min(axis=0)
        self.maximum = vertices_M.max(axis=0)
        start = self.model.mesh_faceadr[mesh]
        n = self.model.mesh_facenum[mesh]
        self.faces = np.array(self.model.mesh_face[start:start+n],dtype=int)
        # Preload an XY spatial index of the mesh. All later elevation queries
        # interpolate these fixed triangles; they do not fire another sensor
        # ray or read the spacecraft pose. Preserve the actual mesh topology.
        triangles=vertices_M[self.faces]
        self._triangle_origin=triangles[:,0]
        self._triangle_u=triangles[:,1]-triangles[:,0]
        self._triangle_v=triangles[:,2]-triangles[:,0]
        self._det=(self._triangle_u[:,0]*self._triangle_v[:,1]
                   -self._triangle_u[:,1]*self._triangle_v[:,0])
        self._cell_m=50.
        self._cells={}
        lower=np.floor((triangles[:,:,:2].min(axis=1)-self.minimum[:2])/self._cell_m).astype(int)
        upper=np.floor((triangles[:,:,:2].max(axis=1)-self.minimum[:2])/self._cell_m).astype(int)
        for i in np.flatnonzero(np.abs(self._det)>1e-12):
            for ix in range(lower[i,0],upper[i,0]+1):
                for iy in range(lower[i,1],upper[i,1]+1):
                    self._cells.setdefault((ix,iy),[]).append(i)
        self._cells={key:np.asarray(value,dtype=int) for key,value in self._cells.items()}

    @staticmethod
    def to_map(r_N):
        return (np.asarray(r_N)-TERRAIN_POSITION_BSK) @ MAP_TO_N

    @staticmethod
    def to_inertial(r_M):
        """Local map metres -> Moon-centred inertial metres, for this static map.

        Column-vector form: r_N = origin_N + MAP_TO_N @ r_M. NumPy uses
        row vectors here so a batch of positions also works. This is a fixed
        rigid transform, not latitude/longitude conversion or Moon rotation.
        Local +X -> inertial +X; local +Y -> inertial +Z; local +Z -> inertial -Y.
        """
        return np.asarray(r_M) @ MAP_TO_N.T + TERRAIN_POSITION_BSK

    def height(self, xy):
        """Preloaded DEM elevation at map XY, not vehicle height or a range.

        Barycentric interpolation on the compiled collision-mesh triangles.
        The uppermost surface is used if multiple triangles cover the same XY.
        """
        xy = np.asarray(xy,dtype=float)
        if np.any(xy < self.minimum[:2]) or np.any(xy > self.maximum[:2]):
            return None
        cell=tuple(np.floor((xy-self.minimum[:2])/self._cell_m).astype(int))
        ids=self._cells.get(cell)
        if ids is None:
            return None
        delta=xy-self._triangle_origin[ids,:2]
        u=self._triangle_u[ids];v=self._triangle_v[ids]
        a=(delta[:,0]*v[:,1]-delta[:,1]*v[:,0])/self._det[ids]
        b=(u[:,0]*delta[:,1]-u[:,1]*delta[:,0])/self._det[ids]
        inside=(a>=-1e-9)&(b>=-1e-9)&(a+b<=1+1e-9)
        if not inside.any():
            return None
        heights=self._triangle_origin[ids,2]+a*u[:,2]+b*v[:,2]
        return float(np.max(heights[inside]))

    def clearance(self, r_N):
        """Map-relative clearance of a supplied position.

        Planning/assessment use simulated positions; the engine cutoff uses
        estimated foot positions. Navigation Z comes from laser reconstruction
        or inertial propagation, never from this function.
        """
        p = self.to_map(r_N)
        h = self.height(p[:2])
        return None if h is None else float(p[2]-h)

    def assess_site(self, xy, radius_m=2., spacing_m=1., max_slope_deg=10., max_roughness_m=.2):
        samples=[]
        for dx in np.arange(-radius_m,radius_m+.5*spacing_m,spacing_m):
            for dy in np.arange(-radius_m,radius_m+.5*spacing_m,spacing_m):
                h=self.height(np.asarray(xy)+[dx,dy])
                if h is None:
                    return dict(mapped=False,acceptable=False,reason='outside map coverage')
                samples.append([dx,dy,h])
        points=np.asarray(samples)
        design=np.column_stack([points[:,:2],np.ones(len(points))])
        fit=np.linalg.lstsq(design,points[:,2],rcond=None)[0]
        slope=float(np.rad2deg(np.arctan(np.linalg.norm(fit[:2]))))
        roughness=float(np.max(np.abs(points[:,2]-design@fit)))
        return dict(mapped=True,acceptable=slope<=max_slope_deg and roughness<=max_roughness_m,
                    slope_deg=slope,roughness_m=roughness,height_m=float(fit[2]),
                    maximum_height_m=float(points[:,2].max()),footprint_radius_m=radius_m)

    def find_site(self, preferred_xy, search_radius_m=150., spacing_m=10., **criteria):
        candidates=sorted((dx*dx+dy*dy,dx,dy)
                          for dx in np.arange(-search_radius_m,search_radius_m+.1,spacing_m)
                          for dy in np.arange(-search_radius_m,search_radius_m+.1,spacing_m)
                          if dx*dx+dy*dy <= search_radius_m**2)
        for _,dx,dy in candidates:
            xy=np.asarray(preferred_xy)+[dx,dy]
            result=self.assess_site(xy,**criteria)
            if result['acceptable']:
                return xy,result
        raise ValueError('No mapped site meets the configured slope/roughness/footprint criteria')

    def export_vizard_obj(self, path):
        """Bake the collision vertices into Vizard's Unity local coordinates.

        Use rotation=[0,0,0], scale=[1,1,1], and the existing terrain anchor.
        Vizard's OBJ custom-model path consumes raw vertices in Unity axes.
        """
        path=Path(path)
        vertices=(self.vertices_N-TERRAIN_POSITION_BSK) @ BSK_TO_UNITY.T
        signature=hashlib.sha256(vertices.tobytes()+self.faces.tobytes()).hexdigest()
        if path.exists() and path.with_suffix('.sha256').exists() and path.with_suffix('.sha256').read_text()==signature:
            return str(path.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            stream.write('# Collision-aligned terrain; Unity local coordinates, metres\n')
            for v in vertices:
                stream.write('v %.9f %.9f %.9f\n'%tuple(v))
            # Right-handed -> left-handed coordinate conversion reverses winding.
            for a,b,c in self.faces:
                stream.write(f'f {a+1} {c+1} {b+1}\n')
        path.with_suffix('.sha256').write_text(signature)
        return str(path.resolve())
