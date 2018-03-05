from database.database import DatabaseTorch

data_path = "/data/scene-segmentation/CamVid/"

folder_str = ['train', 'test', 'val', 'trainval']
folder_str_annot = ['trainannot', 'testannot', 'valannot', 'trainvalannot']

db = DatabaseTorch(data_path, folder_str, folder_str_annot)
data = db(batch_size = 3, shuffle = False, num_workers =20)
