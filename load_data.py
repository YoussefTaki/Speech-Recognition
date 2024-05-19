
from Features import get_features,get_features_aug
from tqdm import tqdm



def get_data(data_path, aug=True):
    if aug:
        X,Y=[],[]
    for path,emotion,index in tqdm (zip(data_path.Path,data_path.Emotions,range(data_path.Path.shape[0]))):
        features=get_features_aug(path)
        if index%500==0:
            print(f'{index} audio has been processed')
        for i in features:
            X.append(features)
            Y.append(emotion)
        print('Done')
    
    else:
        X,Y=[],[]
        for path,emotion,index in tqdm (zip(data_path.Path,data_path.Emotions,range(data_path.Path.shape[0]))):
            features=get_features(path)
    
        X.append(features)
        Y.append(emotion)
        print('Done')
        
    
    return X,Y