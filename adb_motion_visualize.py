import nimble
import joblib
import pickle
import trimesh
import pyvista
import os



custom_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim("./osim/Rajagopal2015_passiveCal_hipAbdMoved_noArms.osim")
skeleton: nimble.dynamics.Skeleton = custom_opensim.skeleton

test_list=os.listdir("./data")
for filename in test_list:
    data = joblib.load(open(filename, 'rb'))
    base_name=os.path.splitext(filename)[0]
    dirs=os.path.join('figure',basename)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    motion_dirs=os.path.join(dirs,basename)
    if not os.path.exists(motion_dirs):
        os.makedirs(motion_dirs)

    qpos=data['qpos']

    for frame in tqdm.tqdm(range(qpos.shape[0])):

        skeleton.setPositions(qpos[frame])

        def pyvista_read_vtp(fn):
            
            if '.ply' in fn:
                fn = fn.split('.ply')[0]
            
            reader = pyvista.get_reader(fn)
            mesh = reader.read()
            mesh = mesh.triangulate()
            
            faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:] 
            tmesh = trimesh.Trimesh(mesh.points, faces_as_array) 
            return tmesh
        objects = []

        for k,b in enumerate(skeleton.getBodyNodes()):
            n = b.getNumShapeNodes()
            for i in range(n):
                s = b.getShapeNode(i)
                name = s.getName().split('_ShapeNode')[0]
                shape = s.getShape()
                try:

                    mesh = pyvista_read_vtp(shape.getMeshPath())
                    mesh = mesh.apply_transform(s.getWorldTransform().matrix())

                    mesh.visual.face_colors=[135, 206, 250, 100]
                    objects.append(mesh)
                except Exception as e:
                    print(e, shape.getMeshPath(), name, s.getName())

        scene = trimesh.Scene()
        for o in objects:
            scene.add_geometry(o)
        scene.set_camera(angles=(-pi/8,pi/2+pi/4,0),distance=2.5) 
        motion_picture=scene.save_image(visible=False)   
        from PIL import Image
        rendered = Image.open(trimesh.util.wrap_as_stream(motion_picture))






