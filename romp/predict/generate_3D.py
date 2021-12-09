'''
Modified image.py to add 2D-to-3D color mapping on the generated .obj mesh file.

The usage is the same:
python -m romp.predict.generate_3D --inputs=3D-Person-reID/3DMarket/split/market_demo --output_dir=demo/image_results

mark: output_dir is useless. The .obj file will be saved into ROMP/3D-Person-reID/3DMarket/split/<folderName>

-- Zhu Jiayin
'''

import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/image.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .base_predictor import *
import constants
import glob
from utils.util import collect_image_list

class Image_processor(Predictor):
    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    @torch.no_grad()
    def run(self, image_folder, tracker=None):
        print('Processing {}, saving to {}'.format(image_folder, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = self.output_dir 
        counter = Time_counter(thresh=1)

        if self.show_mesh_stand_on_image:
            from visualization.vedo_visualizer import Vedo_visualizer
            visualizer = Vedo_visualizer()
            stand_on_imgs_frames = []

        file_list = collect_image_list(image_folder=image_folder, collect_subdirs=self.collect_subdirs, img_exts=constants.img_exts)
        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
        counter.start()
        results_all = {}
        for test_iter,meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
                
            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], \
                    show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)

                for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['mesh_rendering_orgimgs']['figs']):
                    save_name = os.path.join(self.output_dir, os.path.basename(img_name))
                    cv2.imwrite(save_name, cv2.cvtColor(mesh_rendering_orgimg, cv2.COLOR_RGB2BGR))

            if self.show_mesh_stand_on_image:
                stand_on_imgs = visualizer.plot_multi_meshes_batch(outputs['verts'], outputs['params']['cam'], outputs['meta_data'], \
                    outputs['reorganize_idx'].cpu().numpy(), interactive_show=self.interactive_vis)
                stand_on_imgs_frames += stand_on_imgs

            if self.save_mesh:
                # save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
                self.save_meshes(reorganize_idx, outputs, self.smpl_faces)
            
            if test_iter%8==0:
                print('Processed {} / {} images'.format(test_iter * self.val_batch_size, len(internet_loader.dataset)))
            counter.start()
            results_all.update(results)
        return results_all

    def save_meshes(self, reorganize_idx, outputs, faces):
        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx==vid)[0]
            img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
            for subject_idx, batch_idx in enumerate(verts_vids):
                img = outputs['meta_data']['image'][batch_idx]
                self.save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                        faces, outputs['params']['cam'][batch_idx].cpu().numpy().astype(np.float16), img, img_path.replace('.jpg', '').replace('.png', '') + '_' + str(subject_idx))

    def save_obj(self, verts, faces, cam, img, img_path):
        camera = np.reshape(cam, [1, 3]) # 1*3. [[scale, tranlation_x, tranlation_y]]

        w, h, _ = img.shape # 128*64 = 8192; 512*512?
        imgsize = max(w, h)
        # project to 2D
        # verts (6890, 3) (pixel, coordination)
        # print(verts)
        # print(verts.shape) # (6890, 3)
        vert_3d = verts
        vert_2d = verts[:, :2] + camera[:, 1:]  # [[vx + camx, vy + camy], ]
        vert_2d = vert_2d * camera[0, 0] # broad to each pixel, times scale

        # img_copy = img.copy()
        obj_mesh_name = '3D-Person-reID/3DMarket/%s/%s/%s.obj' % (
            'split', os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))
        store_dir = os.path.dirname(obj_mesh_name)
        if not os.path.exists(os.path.dirname(store_dir)):
            os.mkdir(os.path.dirname(store_dir))
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)

        foreground_index_2d = np.zeros((w, h)) + 99999  # shape: w*h, value: 99999
        foreground_value_2d = np.zeros((w, h)) + 99999
        background = np.zeros((w,h))
        index = 6891
        with open(obj_mesh_name, 'w') as fp:
            w, h, _ = img.shape
            imgsize = max(w, h)
            # Decide Forground
            for i in range(vert_2d.shape[0]): # i is pixel index
                v2 = vert_2d[i, :] # x, y
                v3 = vert_3d[i, :] # x, y, z
                z = v3[2]
                x = int(round((v2[1] + 1) * 0.5 * imgsize))
                y = int(round((v2[0] + 1) * 0.5 * imgsize))
                if w < h:
                    x = int(round(x - h / 2 + w / 2))
                else:
                    y = int(round(y - w / 2 + h / 2))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                if z < foreground_value_2d[x, y]:
                    foreground_index_2d[x, y] = i
                    foreground_value_2d[x, y] = z # most foreground z
            # s smooth
            z_max = max(vert_3d[:, 2]) - min(vert_3d[:, 2])
            for t in range(10):
                for i in range(1, w - 1):
                    for j in range(1, h - 1):
                        center = foreground_value_2d[i, j]
                        if foreground_index_2d[i - 1, j] != 999999 and foreground_value_2d[i - 1, j] > center + 0.05:
                            foreground_index_2d[i - 1, j] = 999999
                            foreground_value_2d[i - 1, j] = 999999
                        if foreground_index_2d[i, j - 1] != 999999 and foreground_value_2d[i, j - 1] > center + 0.05:
                            foreground_index_2d[i, j - 1] = 999999
                            foreground_value_2d[i, j - 1] = 999999
            # Draw Color
            for i in range(vert_2d.shape[0]):
                v2 = vert_2d[i, :]
                # v3 = verts[i, :]  # v3 is vertices' position x,y,z
                v3 = vert_3d[i, :]  # v3 is vertices' position x,y,z
                z = v3[2]
                x = int(round((v2[1] + 1) * 0.5 * imgsize))
                y = int(round((v2[0] + 1) * 0.5 * imgsize))
                if w < h:
                    x = int(round(x - h / 2 + w / 2))
                else:
                    y = int(round(y - w / 2 + h / 2))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                if i == foreground_index_2d[x, y]:
                    # fetch pixel at (x,y) in original image. so (x,y) is the pos
                    # calculated
                    c = img[x, y, :] / 255.0
                else:
                    c = [1, 1, 1]
                    # continue # !!!!!!!!!!!! debug 的血泪

                fp.write('v %f %f %f %f %f %f\n' %
                         (v3[0], v3[1], v3[2], c[0], c[1], c[2]))  # x, y, z, r,g,b
            # 2D to 3D mapping
            for i in range(w):
                for j in range(h):
                    vx, vy = i, j
                    if foreground_index_2d[i, j] < 99999:
                        continue
                    if w < h:
                        vx = vx + h / 2 - w / 2
                    else:
                        vy = vy + w / 2 - h / 2
                    vx = vx / imgsize * 2 - 1
                    vy = vy / imgsize * 2 - 1

                    vy /= camera[0, 0]
                    vy -= camera[:, 1]
                    vx /= camera[0, 0]
                    vx -= camera[:, 2]
                    vz = np.mean(verts[:, 2])
                    c = img[i, j, :] / 255.0
                    fp.write('v %f %f %f %f %f %f\n' %
                             (vy, vx, vz, c[0], c[1], c[2]))
                    background[i, j] = index
                    index += 1

            for f in faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
                # break  # skip for small file
            # count = 0
            for i in range(1, w):
                for j in range(1, h):
                    fp.write( 'f %d %d %d %d\n' % (background[i,j], background[i-1,j] ,background[i,j-1] , background[i-1, j-1]))
            print('Finish', obj_mesh_name)



def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Image_processor(args_set=args_set)
        inputs = args_set.inputs
        if not os.path.exists(inputs):
            print("Didn't find the target directory: {}. \n Running the code on the demo images".format(inputs))
            inputs = os.path.join(processor.demo_dir,'images')
        processor.run(inputs)

if __name__ == '__main__':
    main()
