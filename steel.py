from skimage import transform
import numpy as np
import tensorflow as tf
import math
import cv2


# the function of clustering, which gets the group of candidate center

def clustering(candidate_center,threshold_dis):
    x = candidate_center[:,1]
    y = candidate_center[:,0]
    group_distance = []
    for i in range(len(candidate_center)):
        xpoint, ypoint = x[i], y[i]
        xTemp, yTemp = x, y 
        distance = np.sqrt(pow((xpoint-xTemp),2)+pow((ypoint-yTemp),2)) #Calculating distance between two candidate center
        distance_matrix = np.vstack((np.array(range(len(candidate_center))),distance)) 
        distance_matrix = np.transpose(distance_matrix)   #Transposing calculated distance matrix
        distance_sort = distance_matrix[distance_matrix[:,1].argsort()] 
        distance_sort = np.delete(distance_sort,0,axis = 0)
        thre_matrix = distance_sort[distance_sort[:,1]<=threshold_dis] #Sorting distance matrix and calculating threshold matrix
        thre_point = thre_matrix[:,0]
        thre_point = thre_point.astype(np.int)
        thre_point = thre_point.tolist()
        thre_point.insert(0,i)
        group_distance.append(thre_point) #Updating group distance by adding calculated threshold points
    
    group_clustering = [[]] 
    
    for i in range(len(candidate_center)):    
        m1 = group_distance[i]
        for j in range(len(group_clustering)):
            m2 = group_clustering[j]
            com = set(m2).intersection(set(m1)) #Finding intersection between two sets m1 & m2 from group distance and group clustering respectively
            if len(com) == 0:
                if j == len(group_clustering)-1: #Operation on set union of m1 & m2
                    group_clustering.append(m1)
            else:
                m = set(m1).union(set(m2))
                group_clustering[j] = []
                group_clustering[j] = list(m)
                break
    group_clustering.pop(0)
    
    return group_clustering  #the group of candiate center


# the function of candidate center detect which uses Convolutional Neural Network (ConvNet or CNN)

def detect(imageoriginal,stride):  
    patch_size = 71
    height = np.size(imageoriginal,0) 
    width = np.size(imageoriginal,1)
    imageoriginal = transform.resize(imageoriginal,(height,width))   #Down-sizing of image by passing parameters (dimensional) of original image
    imgs=[]
    coordidate = []
    for i in range(patch_size,height-patch_size,stride):
        for j in range(patch_size,width-patch_size,stride):
            imageoriginal_patch = imageoriginal[int(i-(patch_size-1)/2):int(i+(patch_size-1)/2+1),int(j-(patch_size-1)/2):int(j+(patch_size-1)/2+1),:]
            imgs.append(imageoriginal_patch)
            coordidate.append([i,j])
    data = np.asarray(imgs,np.float32)
    output =[]
    
    with tf.compat.v1.Session() as sess:
        #Loading model checkpoint file
        save = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
        save.restore(sess,tf.train.latest_checkpoint('model/')) #Updating Checkpoint
        graph = tf.compat.v1.get_default_graph() #using innermost graph, if not then global graph is created
        x = graph.get_tensor_by_name("x:0") #Fetching tensor value
        vol_slice = 5000
        num_slice = math.ceil(np.size(data,0)/vol_slice) # Ceiling integral value
        for i in range(0,num_slice,1):
            if i+1 != num_slice:
                data_temp = data[i*vol_slice:(i+1)*vol_slice]            
            else:
                data_temp = data[i*vol_slice:np.size(data,0)]
                
            feed_dict = {x:data_temp}
            logits = graph.get_tensor_by_name("logits_eval:0")
            classification_result = sess.run(logits,feed_dict)
            output_temp = tf.argmax(classification_result,1).eval() #returns max value of results
            output = np.hstack((output,output_temp))   
    candidate_center = []
    for i in range(len(output)):
        if output[i] == 1:
            candidate_center.append(coordidate[i])            
       
    return np.array(candidate_center)  #the matrix of candidate center



#the function of clustering the final center

def center_clustering(candidate_center,group_clustering):
    final_result = []
    for i in range(len(group_clustering)): 
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:,1] #Calculating center coordinates
        yz = points_coord[:,0]
        x_mean = np.mean(xz)  #Finding mean value along XZ axis
        y_mean = np.mean(yz)  #Finding mean value along YZ axis
        final_result.append([y_mean,x_mean])
    final_result = np.array(final_result)
    final_result = final_result.astype(np.int)
    
    return final_result # the matrix of final center of steel bars


# the function of showing the result, include the result of candidate center, 
# the bounding-box of clustering, the center of clustering

def show_green_box(imageoriginal,candidate_center,group_clustering):
    cv2.namedWindow('green box')   #Creating output window
    global counter #Declaring the counter variable
    for i in range(len(candidate_center)):
        cv2.circle(imageoriginal,(candidate_center[i,1],candidate_center[i,0]),2,(0,0,255),-1)  #Drawing candidate centres over the image
    for i in range(len(group_clustering)):
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:,1]  #Center coordinates are plotted
        yz = points_coord[:,0]
        cv2.rectangle(imageoriginal,(min(xz)-5,min(yz)-5),(max(xz)+5,max(yz)+5),(0,255,0)) #Drawing rectangle green box around the centers
        
        counter=counter+1  #Counter is incremented
        
    cv2.imshow('green box',imageoriginal)  #Output image is displayed inside previously created window

def show_original_red_point(imageoriginal,candidate_center):
    cv2.namedWindow('original red point')  #Creating output window  
    for i in range(len(candidate_center)):
        cv2.circle(imageoriginal,(candidate_center[i,1],candidate_center[i,0]),2,(0,0,255),-1) #Drawing candidate centers of each bars in the image
    cv2.imshow('original red point',imageoriginal) #Output image is displayed inside previously created window

def show_clustering_red_point(imageoriginal,center_cluster):
    cv2.namedWindow('clustering red point')    #Creating output window
    for i in range(len(center_cluster)):
        cv2.circle(imageoriginal,(center_cluster[i,1],center_cluster[i,0]),2,(0,0,255),-1) #Drawing center of each bars in the image
    cv2.imshow('clustering red point',imageoriginal)     #Output image is displayed inside previously created window
        
if __name__ == "__main__":
    counter=0   #Initializing the counter variable
    while(True):
        
        imageoriginal = cv2.imread('image/test.bmp')  #Reading the image (input)
        
        
        stride = 6 #the parameter of slide window stride
        
        candidate_center = detect(imageoriginal,stride)   
        
       
        distance_threshold = 20  # the parameter of distance clustering threshold
        group_clustering = clustering(candidate_center,distance_threshold)       
        center_cluster = center_clustering(candidate_center,group_clustering)
        
        #show the result
        
        show_original_red_point(imageoriginal,candidate_center)
        
        imageoriginal = cv2.imread('image/test.bmp')
        show_clustering_red_point(imageoriginal,center_cluster)
        
        imageoriginal = cv2.imread('image/test.bmp')
        show_green_box(imageoriginal,candidate_center,group_clustering) 
        
        key = cv2.waitKey(0)
        if key == ord('q'):  #Press "Q" to quit the program
            break
    cv2.destroyAllWindows
    print("total no. of bars= ",counter) #Returns the calculated number of bars