"""
	Methods for finding an automatic initial estimate of the model 

"""

import os
import cv2
import colorsys
import numpy as np
import Plots
import cPickle as pickle
import landmarks
import matplotlib.pyplot as plt
import split_jaws
import task2 
from utils import Timer

jaw_split = split_jaws.Path([])
plot_bbox_dist = False
plot_app_models = False
plot_finding_bbox = False
plot_autoinit_bbox = False
plot_autoinit_lms = False
save_plots = False
save_dir = ""

def get_estimate(asm_list,incisor_list, test_img_idx, show_bbox_dist=False, show_app_models=False, \
                 show_finding_bbox=False, show_autoinit_bbox=False, show_autoinit_lms=False, save=False):
    """
    Finds an initial estimate for all the incisors in the incisor_list
    """
    global plot_bbox_dist
    global plot_app_models
    global plot_finding_bbox
    global plot_autoinit_bbox
    global plot_autoinit_lms
    global save_plots
    global save_dir
    global jaw_split
    
    plot_bbox_dist = show_bbox_dist
    plot_app_models = show_app_models
    plot_finding_bbox = show_finding_bbox
    plot_autoinit_bbox = show_autoinit_bbox
    plot_autoinit_lms = show_autoinit_lms
    save_plots = save
    save_dir = "Plots/auto_init/test_img_%02d/" %(test_img_idx)
 
    with Timer("Finding Initial Estimate automatically"):
        
        if any(incisor < 5 for incisor in incisor_list): # upper incisor
            is_upper= True
            with Timer("..for upper incisors", dots="...."):
                [(w1U, h1U), (w2U, h2U)] = get_big_bbox(is_upper, test_img_idx)
            
        if any(incisor > 4 for incisor in incisor_list): # lower incisor
            is_upper= False
            with Timer("..for lower incisors", dots="...."):
                [(w1L, h1L), (w2L, h2L)] = get_big_bbox(is_upper, test_img_idx)        
    
    print("") # just for elegant printing on screen   
        
    init_list = []
    test_img = task2.load([test_img_idx])[0]
    img_org = test_img.copy()
    test_img = task2.enhance(test_img, skip_amf=True)
        
    for index,incisor in enumerate(incisor_list): 
        # Assume all teeth have more or less the same width
        
        if incisor < 5:
            ind = incisor
            bbox = [(w1U +(ind-1)*(w2U-w1U)/4, h1U), (w1U +(ind)*(w2U-w1U)/4, h2U)]
        else:
            ind = incisor - 4
            bbox = [(w1L +(ind-1)*(w2L-w1L)/4, h1L), (w1L +(ind)*(w2L-w1L)/4, h2L)]
            
        center = np.mean(bbox, axis=0)
        
        Plots.plot_autoinit(test_img, jaw_split, lowest_error_bbox=bbox, directory=save_dir, \
                            title="initial_estimate_bbox_incisor_%d" %(incisor), wait=True, \
                            show=plot_autoinit_bbox, save=False)#save=save_plots

        init = asm_list[index].sm.mean_shape.scale_to_bbox(bbox).translate(center)
        Plots.plot_landmarks_on_image([init], img_org, directory=save_dir, \
                                      title="initial_estimate_lms_incisor_%d" %(incisor), \
                                      show=plot_autoinit_lms, save=False, color=(0,255,0))#save=save_plots
        
        init_list.append(init)
        
    return init_list

def get_big_bbox(is_upper, test_img_idx):
    """
    Finds the bounding box surrounding all the four upper(or lower) incisors
    """
    
    bbox_list = extract_roi_for_appModel(is_upper, test_img_idx)
    
    directory = "Autoinit_params/test_img_%02d/" %(test_img_idx)
    filename = "upper_incisor.model" if is_upper else "lower_incisor.model"

    if os.path.exists(directory+filename):
        with file(directory+filename, 'rb') as f:
            [smallImages, [def_width, def_height, search_region]] = pickle.load(f)    
    else:
        [smallImages, [def_width, def_height, search_region]] = \
                                get_data_for_auto_init(is_upper, bbox_list, test_img_idx)
        save_file([smallImages, [def_width, def_height, search_region]], directory, filename)
    
    [_, evecs, mean] = pca(smallImages, 5)
    
    global jaw_split
    global plot_app_models
    global save_plots
    global save_dir
    
    #Visualize the appearance model
    app_model = np.hstack( (mean.reshape(def_height,def_width), \
                                  normalize(evecs[:,0].reshape(def_height,def_width)), \
                                  normalize(evecs[:,1].reshape(def_height,def_width)), \
                                  normalize(evecs[:,2].reshape(def_height,def_width))) \
                                ).astype(np.uint8)
    if plot_app_models:
        cv2.imshow('app_model', app_model)
        cv2.waitKey(0)
    
    if save_plots:
        title = "upper_incisors" if is_upper else "lower_incisors"
        save_image(app_model, "appearance_model_"+title+".png", save_dir)
    
    test_img = task2.load([test_img_idx])[0]
    test_img = task2.enhance(test_img, skip_amf=True)
    [(a, b), (c, d)] = find_bbox(mean, evecs, test_img, def_width, def_height, is_upper, \
                                     jaw_split, search_region)
    
    return [(a, b), (c, d)]


def extract_roi_for_appModel(is_upper, test_img_idx):
    """
    Extracts the region of interest (bounding box) surrounding the four upper(or lower) incisors
    """
    
    bbox_list = []
    train_idx = range(1,15)
    train_idx.remove(test_img_idx)
    
    for example_nr in train_idx:
        lms = landmarks.load_all_incisors_of_example(example_nr)
        img = cv2.imread('Data/Radiographs/'+str(example_nr).zfill(2)+'.tif')
        if is_upper:
            bbox = Plots.draw_bbox(img, lms[0:4],show=False,return_bbox=True)
        else:
            bbox = Plots.draw_bbox(img, lms[4:8],show=False,return_bbox=True)
        bbox_list.append(bbox)
    
    return bbox_list


def get_data_for_auto_init(is_upper, bbox_list, test_img_idx):

    img = cv2.imread('Data/Radiographs/'+str(test_img_idx).zfill(2)+'.tif')
    [mean_bbox, search_region] = get_parameters(is_upper, bbox_list, img)

    def_width = abs(mean_bbox[0] - mean_bbox[2])
    def_height = abs(mean_bbox[1] - mean_bbox[3])
    
    smallImages = np.zeros((13, def_width * def_height)) # building model excluding test image
    
#    # can load preprocessed images directly
#    radiographs = task2.load(preprocessed=True)
    
    # or can load and then preprocess
    radiographs = task2.load(preprocessed=False) 
    del radiographs[test_img_idx-1] # deleteing test index
    
    # skip_amf = without median filter
    radiographs = [task2.enhance(radiograph, skip_amf=True) for radiograph in radiographs] 
    
    for ind, radiograph in enumerate(radiographs):
        [x1, y1, x2, y2] = bbox_list[ind]
        cutImage = radiograph[y1:y2, x1:x2]
        result = cv2.resize(cutImage, (def_width, def_height), interpolation=cv2.INTER_NEAREST)
        smallImages[ind] = result.flatten()
    
    return [smallImages, [def_width, def_height, search_region]]


def get_parameters(is_upper, bbox_list, img):
    """
    Computes the parameters required for auto_init
    """
    img_for_split = img.copy()
    colors = get_colors(len(bbox_list))
    meanx1 = 0
    meany1 = 0
    meanx2 = 0
    meany2 = 0
    width_list = []
    height_list = []
    w = img.shape[1]

    for ind, bbox in enumerate(bbox_list):
        cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]),colors[ind],2)    
        meanx1 += bbox[0]
        meany1 += bbox[1]
        meanx2 += bbox[2]
        meany2 += bbox[3]
        width_list.append(bbox[0])
        width_list.append(bbox[2])
        height_list.append(abs(bbox[1] - bbox[3]))

    mean_bbox = [meanx1/(ind+1), meany1/(ind+1), meanx2/(ind+1), meany2/(ind+1)]    
       
#    Plotting bbox of training instances on test image
#    cv2.rectangle(img,(mean_bbox[0], mean_bbox[1]),(mean_bbox[2], mean_bbox[3]),(0,0,0),3)
#    Plots.show_image(img, "contour") 

#==============================================================================
#     # subplots of height and width distribution(from w/2)
#     # These plots were mainly used to decide upon the parameters of search region

#     f = plt.figure(1)
#     plt.scatter(range(1,len(width_list)+1), width_list)
#     plt.plot([1,len(width_list)+1],[w/2, w/2])   
#     plt.xlabel("Extreme points of bbox from w/2")
#     plt.ylabel("Distance from w/2")
#     
#     g = plt.figure(2)
#     plt.plot(range(1,len(height_list)+1), height_list)
#     plt.xlabel("Training radiographs no.")  
#     plt.ylabel("Height of bboxes")
#     
#     global plot_bbox_dist
#     global save_plots
#     global save_dir
#     
#     if plot_bbox_dist: 
#         plt.show()          
#         
#     if save_plots:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         filename = "upper_incisors_" if is_upper else "lower_incisors_"
#         f.savefig(save_dir+filename+"width_dist.png")
#         g.savefig(save_dir+filename+"height_dist.png")
#         plt.close('all')
#==============================================================================
        
    global jaw_split
    jaw_split = split_jaws.get_split(img_for_split, 50, False)
    
    # Search region parameters
    w1 = int( min(width_list) - (0.2*(w/2 - min(width_list))) )
    w2 = int( max(width_list) + (0.2*(max(width_list) - w/2)) )
    if is_upper:        
        h1 = int( (np.max(jaw_split.get_part(w1, w2), axis=0)[1]) - 1.1*(max(height_list)) )
        h2 = int(np.max(jaw_split.get_part(w1, w2), axis=0)[1])
    else:
        h1 = int(np.min(jaw_split.get_part(w1, w2), axis=0)[1])
        h2 = int( (np.min(jaw_split.get_part(w1, w2), axis=0)[1]) + (max(height_list)) )
        
    search_region = [(w1, h1), (w2, h2)]
    
#    cv2.rectangle(img,search_region[0], search_region[1],(0,255,0),4)
#    Plots.show_image(img, "contour")
#    cv2.rectangle(img_for_split,(mean_bbox[0], mean_bbox[1]),(mean_bbox[2], mean_bbox[3]),(0,0,0),3)
#    Plots.show_image(img_for_split, "contour")
    
    return [mean_bbox, search_region]


def find_bbox(mean, evecs, test_img, def_width, def_height, is_upper, \
                                     jaw_split, search_region):
    """
    Finds the bounding box inside the search region, with the lowest reconstruction error 
    """
    lowest_error = float("inf")
    lowest_error_bbox = [(-1, -1), (-1, -1)]
    
    global plot_finding_bbox
    global save_plots
    global save_dir
    
    current_window = []
    lowest_error_bbox = []
    for wscale in np.arange(0.8, 1.3, 0.1):
        for hscale in np.arange(0.7, 1.2, 0.1):
            winW = int(def_width * wscale)
            winH = int(def_height * hscale)
            
            for (x, y, current_window) in sliding_window(test_img, search_region, step_size=20, \
                                                        window_size=(winW, winH)):
                if current_window.shape[0] != winH or current_window.shape[1] != winW:
                    continue

                reCut = cv2.resize(current_window, (def_width, def_height))

                X = reCut.flatten()
                Y = project(evecs, X, mean)
                Xacc = reconstruct(evecs, Y, mean)

                error = np.linalg.norm(Xacc - X)
                if error < lowest_error:
                    lowest_error = error
                    lowest_error_bbox = [(x, y), (x + winW, y + winH)]
                        
                current_window = [(x, y), (x + winW, y + winH)]
                
                sub_dir = "upper_incisors/" if is_upper else "lower_incisors/"
                directory = save_dir+"finding_bboxes/"+sub_dir
                Plots.plot_autoinit(test_img, jaw_split, current_window, search_region, \
                                    lowest_error_bbox, directory=directory, \
                                    title="wscale="+str(wscale)+" hscale="+str(hscale), \
                                    wait=False, show=plot_finding_bbox, save=False)
   
    # Plot of final chosen window
    title= "upper" if is_upper else "lower"
    if plot_finding_bbox or save_plots:
        Plots.plot_autoinit(test_img, jaw_split, current_window, search_region, \
                            lowest_error_bbox, directory=save_dir, \
                        title="initial_estimate_bbox_%s" %(title), wait=False, \
                        show=plot_finding_bbox, save=save_plots)

    return lowest_error_bbox

def sliding_window(image, search_region, step_size, window_size):
    """
    Returns a sliding window object
    """
    for y in range(search_region[0][1], search_region[1][1] - window_size[1], step_size) + \
                [search_region[1][1] - window_size[1]]:
        for x in range(search_region[0][0], search_region[1][0] - window_size[0], step_size) + \
                [search_region[1][0] - window_size[0]]:
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def project(W, X, mu):
    """Project X on the space spanned by the vectors in W.
    mu is the average image.
    """
    return np.dot(X - mu.T, W)


def reconstruct(W, Y, mu):
    """Reconstruct an image based on its PCA-coefficients Y, the evecs W
    and the average mu.
    """
    return np.dot(Y, W.T) + mu.T


def pca(X, nb_components=0):
    """Do a PCA analysis on X
    Args:
        X: np.array containing the samples
            shape = (nb samples, nb dimensions of each sample)
        nb_components: the nb components we're interested in
    Returns:
        The ``nb_components`` largest evals and evecs of the covariance matrix and
        the average sample.
    """
    [n, d] = X.shape
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    mu = np.average(X, axis=0)
    X -= mu.transpose()

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
    eigenvectors = np.dot(np.transpose(X), eigenvectors)

    eig = zip(eigenvalues, np.transpose(eigenvectors))
    eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                         x[1] / np.linalg.norm(x[1])), eig)

    eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
    eig = eig[:nb_components]

    eigenvalues, eigenvectors = map(np.array, zip(*eig))

    return eigenvalues, np.transpose(eigenvectors), mu


def normalize(img):
    """Normalize an image such that it min=0 , max=255 and type is np.uint8
    """
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)
    
def save_file(X, directory, filename):

    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    fn = os.path.join(directory, filename)
    with open(fn, 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    f.close()        

def save_image(img, title="plot", directory="Plot/auto_init/"):
    """
    saves an image in a given directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)    
    cv2.imwrite(directory+title, img)
  
def get_colors(num_colors):
    """
    Returns a list with num_colors different colors.
    Source: http://stackoverflow.com/a/9701141
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in colors]