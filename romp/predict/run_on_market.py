'''
Modified image.py to add 2D-to-3D color mapping on the generated .obj mesh file.

The usage is the same:
python -m romp.predict.generate_3D --inputs=3D-Person-reID/3DMarket/split/market_demo --output_dir=demo/image_results

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
from PIL import Image

class Image_processor(Predictor):
    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    @torch.no_grad()
    def run(self, image_folder, splits=['gallery','multi-query','query','train','train_all','val'], tracker=None):
        print('Processing {}, saving to {}'.format(image_folder, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        counter = Time_counter(thresh=1)

        for split in splits:
            input_dir = image_folder + '/' + split
            output_dir = self.output_dir + '/' + split
            print('Split {} --------\nProcessing {}, saving to {}'.format(split, input_dir, output_dir))
            os.makedirs(output_dir, exist_ok=True)

            file_list = collect_image_list(image_folder=input_dir, collect_subdirs=True, img_exts=constants.img_exts)
            internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
            counter.start()
            results_all = {}
            for test_iter,meta_data in enumerate(internet_loader):
                outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                counter.count(self.val_batch_size)
                results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

                if self.save_mesh:
                    # save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
                    self.save_meshes(reorganize_idx, outputs, self.smpl_faces, output_dir)

                if test_iter%8==0:
                    print('Processed {} / {} images'.format(test_iter * self.val_batch_size, len(internet_loader.dataset)))
                counter.start()
                results_all.update(results)
        return results_all

    def save_meshes(self, reorganize_idx, outputs, faces, output_dir):
        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx==vid)[0]
            img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
            for subject_idx, batch_idx in enumerate(verts_vids):
                img = outputs['meta_data']['image'][batch_idx]

                # crop padding of the img

                # print('idx, sub_idx, batch_idx')
                # print(idx, subject_idx, batch_idx)
                offsets = outputs['meta_data']['offsets'].cpu().numpy().astype(np.int)[batch_idx]
                offsets = np.reshape(offsets, (1, 10))
                img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
                # org_imge = cv2.imread(img_path)
                org_imge = np.array(Image.open(img_path))

                self.save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), faces, org_imge, img_path.replace('.jpg', '').replace('.png', ''),  output_dir )

    def save_obj(self, verts, faces, img, img_path, output_dir):
        # verts (6890, 3) (pixel, coordination)
        obj_mesh_name = '%s/%s/%s.obj' % ( output_dir, \
             os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))
        store_dir = os.path.dirname(obj_mesh_name)
        if not os.path.exists(os.path.dirname(store_dir)):
            os.mkdir(os.path.dirname(store_dir))
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)

        # index = 6891
        with open(obj_mesh_name, 'w') as fp:
            for i in range(verts.shape[0]):
                v3 = verts[i, :]  # v3 is vertices' position x,y,z

                fp.write('v %f %f %f\n' % (v3[0], v3[1], v3[2]))

            fp.write('f 1 2 3\n')


def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Image_processor(args_set=args_set)
        inputs = args_set.inputs
        processor.run(inputs)

if __name__ == '__main__':
    main()
