#%%
from turbolpc.analysis import arburg_matrix, arburg_warped_matrix # type: ignore
from turbolpc.utils import freqz, arcoeff_warp,arcoeff_to_cep,cep_to_arcoeff # type: ignore
import numpy as np
from scipy.signal import tf2zpk, zpk2tf
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cmath
from transformers import BertTokenizer
from hugtokencraft import editor # type: ignore

#%% module
@dataclass
class LiPCoTdata:
    label: int
    data: float # signal length x signal windows
    Fs: int
    ar_coeff: float # singal windows x AR order
    ar_sigma: float # singal windows x 1
    ar_feature: float # singal windows x totl features
    token_ids: int = -1 # singal windows x 1
    tokenized: str =''

def calculate_ar(dataset,order,w_factor,case_id):
    lkdataset=[]
    for edata in dataset:
        Fs=edata.Fs
        if w_factor is None:
            ar_coeffs,ar_sigmas,_=arburg_matrix(edata.data, order)
        else:
            ar_coeffs,ar_sigmas,_=arburg_warped_matrix(edata.data,order,w_factor)
            # for using up-warped coeff as features
            """
            for i in range(ar_coeffs.shape[1]):
                ar_coeffs[:,i]=arcoeff_warp(a=ar_coeffs[:,i],warp_factor=w_factor,task="unwarp")
            """
        ar_coeffs= np.real(ar_coeffs.T) # singal windows x AR order
        ar_features=ar_feature_calculation(ar_coeffs,ar_sigmas,Fs,case_id) # singal windows x AR order
        ldata=LiPCoTdata(label=edata.label,data=edata.data,Fs=edata.Fs,ar_coeff=ar_coeffs,ar_sigma=ar_sigmas,ar_feature=ar_features)
        lkdataset.append(ldata)
    return lkdataset

# helper function for calculate_ar which calculates features
def ar_feature_calculation(ar_features,ar_sigmas,Fs,case_id):
    """
    input: ar_feature=obs x feature
            ar_sigmas=1 x obs
    """

    if case_id==1: # method 1
        total_feature=2*ar_features.shape[1]+1
    else: # method 2,3
        total_feature=ar_features.shape[1]+1


    new_ar_features = np.zeros((ar_features.shape[0], total_feature))
    #new_ar_features = np.zeros((ar_features.shape[0], total_feature),dtype=complex) # method 4
    for i in range(ar_features.shape[0]):
        row=ar_features[i,:].copy()
        sigma=ar_sigmas[i].copy()
        
        # method 1
        if case_id==1:
            """
            Use radius and angle of all AR poles
            w/o angle normalized and sigma added as another feature
            """
            # Insert 1 at the beginning since we removed it during arburg
            a=np.insert(row, 0, 1)
            b=np.zeros_like(a)
            b[0]=1
            #get poles
            z, p, k = tf2zpk(b, a)

            p_angles=Fs*np.abs(np.angle(p.copy()))/(2*np.pi)
            p=p[np.argsort(p_angles)] # reorder poles by increasing frequency. Adding these doesn't change things much
            p_rad=-2*np.log10(1-np.abs(p.copy()))
            #p_rad=np.abs(p)
            p_ang=Fs*np.angle(p.copy())/(2*np.pi)
            curr_feature=np.concatenate((p_rad,p_ang)) # angle value 0 to 1 but conjugate have same angle
            curr_feature=np.append(curr_feature,np.log10(sigma))         


        #method 2
        if case_id==2:
            """
            Use AR coefficients as features 
            """
            curr_feature=np.append(row,np.log10(sigma))

        #method 3
        if case_id==3:
            """
            Convert linear prediction coefficents (LPC) to cepstral coefficients
            but we can calculate from the poles as well (TODO?)
            """

            curr_feature=arcoeff_to_cep(row,sigma,total_feature)
            # scale: n^.5*c_n for making euclidian compatible to cepstral distance calculation
            curr_feature= np.sqrt(np.insert(np.linspace(1,total_feature-1,total_feature-1),0,1))*curr_feature
            ##curr_feature= curr_feature[1:]


        #method 4
            """
            TODO: Use actual distance metric with dbscan/optics/Agglomerative?
            """
            #curr_feature=p
            ##curr_feature=np.concatenate((np.real(p),np.imag(p)))

        new_ar_features[i,:]=curr_feature.copy()
    return new_ar_features
# helper function to get ar coeff from features
def ar_feature_to_arcoef(ar_feature,lk_model):
    case_id=lk_model['method']
    order=lk_model['order']
    Fs=lk_model['Fs']
    """
    takes ar_feature and outputs ar coeff [a1,a2,...]
    """
    if case_id==2:
        # method 2
        sigma=ar_feature[-1]
        sigma=10**sigma
        a=ar_feature[:-1]

    if case_id==3:
        # method 3
        total_feature=ar_feature.shape[0]
        sigma=ar_feature[0].copy()
        sigma=np.exp(sigma)
        curr_feature= 1/np.sqrt(np.insert(np.linspace(1,total_feature-1,total_feature-1),0,1))*ar_feature.copy()
        a=cep_to_arcoeff(curr_feature,order)

    if case_id==1:
        # method 1
        #order=int(myTs.order/2) # method 1.2, 1.3
        order=int(order)
        p_rad=ar_feature[:order].copy()
        p_angle=ar_feature[order:].copy()
        p_angle=p_angle[:-1]# ignore sigma
        p_rad=np.array(p_rad)
        p_angle=np.array(p_angle)
        p_angle=(2*np.pi)*p_angle/Fs
        p_rad=1-10**(p_rad/(-2))
        k=1#10**ar_f[-1]
        p=np.array(p_rad,dtype=complex)
        for i in range(len(p)):
            p[i]=cmath.rect(p_rad[i], p_angle[i]) # error
        z=0*p
        b,a=zpk2tf(z, p, k)
        a=a[1:] # exclude the first 1
        sigma=ar_feature[-1]
        sigma=10**sigma

    return a,sigma

# get ar coeff of cluster centers
def cluster_to_arcoef(lk_model):
    scaler=lk_model['kmeans_model']['scaler']
    cluster_features=lk_model['kmeans_model']['TokenMap'].values()
    cluster_features=np.array(list(cluster_features))
    cluster_arcoeff=[]
    for i in range(cluster_features.shape[0]):
        c_feature=cluster_features[i,:]
        c_feature=scaler.inverse_transform(c_feature.reshape(1,-1))[0]
        ar_coeff,ar_sigma=ar_feature_to_arcoef(c_feature,lk_model)
        cluster_arcoeff.append((ar_coeff,ar_sigma))
    return cluster_arcoeff

# collects the features for signal windows into an array
def collect_features(lkdataset):
    for i,lkdata in enumerate(lkdataset):
        if i==0:
            feature_array=lkdata.ar_feature
        else:
            feature_array=np.vstack((feature_array,lkdata.ar_feature))
    return feature_array

def cluster_features(feature_array,k):
    scaler=StandardScaler()
    #input: obs x feature
    feature_array=scaler.fit_transform(feature_array)
    # Perform KMeans clustering
    kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=10, verbose=0, random_state=42)
    #kmeans_model = BisectingKMeans(n_clusters=k) # slightly worse token matching perf
    kmeans_model.fit(feature_array)
    cluster_centers=kmeans_model.cluster_centers_ # k x feature
    encoded_token_ids = kmeans_model.labels_  # obs x 1
    encoded_token_ids=encoded_token_ids.astype(int)

    # get cluster ids
    cluster_ids=kmeans_model.predict(cluster_centers).astype(int) # k x 1
    #HashMap of cluster center and cluster ids
    lkTokens={}
    for i in range(cluster_centers.shape[0]):
        lkTokens[cluster_ids[i]]=cluster_centers[i,:]
    
    kmeans_model_data={'model': kmeans_model,'scaler':scaler,'TokenMap': lkTokens}
    return kmeans_model_data, encoded_token_ids

# feature_array: obs x feature number
def encode_features(feature_array,kmeans_model_data):
    scaler=kmeans_model_data['scaler']
    kmeans_model=kmeans_model_data['model']
    feature_array=scaler.transform(feature_array)
    cluster_ids=kmeans_model.predict(feature_array)  
    return cluster_ids.astype(int)

def modify_bert_tokenizer(vocab_length,model_max_length,tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #load BertTokenizer
    initial_vocab_size=len(tokenizer) #check
    print(f"initial vocab size: {initial_vocab_size}")
    
    target_vocab_size=vocab_length #Target vocabulary
    selected_words=editor.get_top_tokens(tokenizer,target_vocab_size)
    modified_tokenizer=editor.reduce_vocabulary(tokenizer,selected_words) #reduce vocabulary
    tokenizer_path=editor.save_tokenizer(modified_tokenizer,tokenizer_path,model_max_length)
    
    modified_tokenizer=editor.load_tokenizer(type(tokenizer),tokenizer_path)
    new_vocab_size=len(modified_tokenizer) # check
    print(f"new vocab size: {new_vocab_size} words")

    return modified_tokenizer, list(selected_words)

def encode_dataset(lkdataset,lk_model):
    kmeans_model_data=lk_model['kmeans_model']
    WordMap=lk_model['WordMap']
    unk_token=lk_model['unk_token']
    case_id=lk_model['method']
    order=lk_model['order']
    w_factor=lk_model['w_factor']

    # get lpc coeff and lpc features
    lkdataset=calculate_ar(lkdataset,order,w_factor,case_id)

    # encode features
    for lkdata in lkdataset:
        feature_array=lkdata.ar_feature
        token_ids=encode_features(feature_array,kmeans_model_data)
        lkdata.token_ids=token_ids
    
        result_strings = []
        for token_id in token_ids:
            word = WordMap.get(token_id,unk_token)  # Use unk_str if token_id is not found in the dictionary
            result_strings.append(word)
        tokenized_string=' '.join(result_strings)
        lkdata.tokenized=tokenized_string
    return lkdataset

def lipkot_train(dataset,lk_settings):
    order=lk_settings['order']
    w_factor=lk_settings['w_factor']
    vocab_length=lk_settings['vocab_length']
    case_id=lk_settings['method']

    # get lpc coeff and lpc features
    lkdataset=calculate_ar(dataset,order,w_factor,case_id)
    # collect all features for kmeans
    feature_array=collect_features(lkdataset)
    # kmeans
    kmeans_model_data, _ =cluster_features(feature_array,vocab_length)
    # edit berttokenizer
    tokenizer_path = lk_settings['tokenizer_path']
    model_max_length=lk_settings['model_max_length']
    modified_tokenizer, selected_words=modify_bert_tokenizer(vocab_length,model_max_length,tokenizer_path)
    # cluster_id and word Map
    WordMap={}
    for i,k in enumerate(kmeans_model_data['TokenMap'].keys()):
        WordMap[k]=selected_words[i]
    # trained_model
    lk_model=lk_settings.copy()
    lk_model['kmeans_model']=kmeans_model_data
    lk_model['WordMap']=WordMap
    lk_model['method']=case_id
    lk_model['Fs']=lkdataset[0].Fs
    return lk_model
    
