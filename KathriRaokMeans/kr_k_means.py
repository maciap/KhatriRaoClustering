import numpy as np 
import random 
from sklearn.preprocessing import StandardScaler
import warnings
from kr_k_means_utils import find_current_centers, compute_loss, compute_centroid_movement
from scipy.spatial.distance import cdist

class KrKMeans(): 
    def __init__(self, D, r1, r2, standardize = True, operator = "product"):
        '''
        KrKMeans implements Khatri-Rao K-means for two sets of protocentroids 
        D: input data (np.ndarray)
        r1: number of protocentroids in the first group (int) 
        r2: number of protocentroids in the second group (int) 
        standardize: whether to compute z-scores or not (bool) 
        '''
        if standardize: 
            scaler = StandardScaler()
            D = scaler.fit_transform(D)
        self.D = D.astype('float32') 
        self.r1 = r1 
        self.r2 = r2 
        self.n, self.m = D.shape
        assert operator in ["product", "sum"], f"operator must be 'sum' or 'product' but is {operator}"
        self.operator = operator

        
      
    
    def kmeans_plus_plus_init_deterministic(self):
        """
        implementation of the KMeans++ initialization method for initializating the centroids
        in the Khatri-Rao clustering setting. Picks centroids deterministically (maximum distance from existing ones). 
    
        Returns:
            B1, B2 (ndarray): Initial position of protocentroids
        """
        warnings.warn("Using ++ initialization",)
        
        centroid = self.D[random.randrange(len(self.D))]
        centroids = [centroid]

        centroid1 =  self.D[random.randrange(len(self.D))]  
        centroid2 = centroid/ centroid1

        B1 = np.zeros((self.r1, self.m))
        B2 = np.zeros((self.r2, self.m))

        B1[0,:] = centroid1
        B2[0,:] = centroid2

        for c1 in range(self.r1):
            for c2 in range(self.r2): 

                # Get the squared distance between that centroid and each sample in the dataset
                squared_distances = np.array([min([np.inner(cd - sample, cd - sample) for cd in centroids]) for sample in self.D])

                centroid = np.argmax(squared_distances)#

                if np.sum(B1[c1,:]) == 0  and np.sum(B2[c2,:]) == 0 :
                    centroid1 = self.D[random.randrange(len(self.D))] 
                    if self.operator == "product": 
                       centroid2 = self.D[centroid] / centroid1
                    else: 
                        centroid2 = self.D[centroid] - centroid1
                    B1[c1,:] = centroid1
                    B2[c2,:] = centroid2
                elif np.sum(B1[c1,:]) != 0  and np.sum(B2[c2,:]) == 0 :                            
                    if self.operator == "product": 
                        B2[c2,:] = self.D[centroid] / B1[c1,:]
                    else: 
                        B2[c2,:] = self.D[centroid] - B1[c1,:]
                elif np.sum(B1[c1,:]) == 0 and np.sum(B2[c2,:]) != 0 :                            
                    if self.operator == "product":                       
                        B1[c1,:] = self.D[centroid] / B2[c2,:]    
                    else: 
                        B1[c1,:] = self.D[centroid] - B2[c2,:]   
                     
                centroids.append( self.D[centroid] )    
                    
                if not np.any(np.all(B1 == 0, axis=1)) and not np.any(np.all(B2 == 0, axis=1)): 
                    break 
        
        return B1, B2




    def kmeans_plus_plus_init_random(self):
        """
        implementation of the KMeans++ initialization method for computing the centroids.
        Picks centroids randomly (with probability proportional to distance from existing ones). 
        
        Returns:
        B1, B2 (ndarray): Initial  semi-centroids
        """
        warnings.warn("Using ++ initialization",)

        centroid = self.D[random.randrange(len(self.D))]
        centroids = [centroid]
        
        centroid1 = self.D[random.randrange(len(self.D))]  #np.random.rand(self.m)
        centroid2 = centroid / centroid1
  
        B1 = np.zeros((self.r1, self.m))
        B2 = np.zeros((self.r2, self.m))
              
        B1[0,:] = centroid1
        B2[0,:] = centroid2
    
        # For each cluster
        for c1 in range(self.r1):
            for c2 in range(self.r2): 
    
                # Get the squared distance between that centroid and each sample in the dataset
                squared_distances = np.array([min([np.inner(cnd - sample,cnd - sample) for cnd in centroids]) for sample in self.D])
        
                # Convert the distances into probabilities that a specific sample could be the center of a new centroid
                proba = squared_distances / squared_distances.sum()
                
                centroid = np.random.choice(len(self.D), size=1, p=proba)[0]
                
                if np.sum(B1[c1,:]) == 0  and np.sum(B2[c2,:]) == 0 :
                    centroid1 = self.D[random.randrange(len(self.D))]   
                    if self.operator == "product": 
                        centroid2 = self.D[centroid] / centroid1
                    else: 
                        centroid2 = self.D[centroid] - centroid1
                    B1[c1,:] = centroid1
                    B2[c2,:] = centroid2
                elif np.sum(B1[c1,:]) != 0  and np.sum(B2[c2,:]) == 0 :      
                    if self.operator == "product": 
                        B2[c2,:] = self.D[centroid] / B1[c1,:]
                    else: 
                        B2[c2,:] = self.D[centroid] - B1[c1,:]
                     
                elif np.sum(B1[c1,:]) == 0  and np.sum(B2[c2,:]) != 0 :      
                    if self.operator == "product":                       
                        B1[c1,:] = self.D[centroid] / B2[c2,:]    
                    else: 
                        B1[c1,:] = self.D[centroid] - B2[c2,:]   
                    
                centroids.append(self.D[centroid])     
                    
                if not np.any(np.all(B1 == 0, axis=1)) and not np.any(np.all(B2 == 0, axis=1)): 
                    break 
    
        return B1, B2
    
        
    
    
    def random_inizialization_B(self): 
        """
        random initialization of protocentroids 

        Returns:
        B1, B2 (ndarray): Initial  protocentroids
        """
        B_1 = np.zeros((self.r1, self.m))
        B_2 = np.zeros((self.r2, self.m))

        # initialize by picking random data points 
        for i in range(self.r1): 
            row_index = np.random.choice(self.n)
            # get the selected row
            selected_row = self.D[row_index]
            #
            B_1[i,:] = selected_row 
            #
            #
        for j in range(self.r2): 
            row_index = np.random.choice(self.n)
            # get the selected row
            selected_row2 = self.D[row_index]
            #
            B_2[j,:] = selected_row2 
            
        return B_1, B_2
    
    
    
    def init_ones(self): 
        """
        initialize protocentroids with ones 

        Returns:
        B1, B2 (ndarray): Initial  protocentroids
        """
        B_1 = np.ones((self.r1, self.m))
        B_2 = np.ones((self.r2, self.m))
        return B_1, B_2
        
    def init_zeros(self): 
        """
        initialize protocentroids with zeros 

        Returns:
        B1, B2 (ndarray): Initial protocentroids
        """
        B_1 = np.zeros((self.r1, self.m))
        B_2 = np.zeros((self.r2, self.m))
        return B_1, B_2
        
    
    
    def find_cluster_membership(self, D): 
        ''' find cluster membership and maps to protocentroid indices
        
        Returns: 
        indices for first of protocentroids (ndarray), indices for second set of protocentroids (ndarray), indices for centroids (ndarray)
         ''' 
        distances = cdist(D, self.allcenters)
        indices_best = np.argmin(distances, axis=1)
        return (indices_best // self.r2).flatten() , (indices_best % self.r2).flatten() , indices_best

    
    def update_A(self, indices, r): 
        '''update cluster membership (assignment) matrix 
        
        Returns:
        A (ndarray): Updated assignment matrix 
        '''
        A = np.zeros((self.n,r)) 
        for i in range(self.n): 
            A[ i, indices[i] ] = 1 
        return A
    
    
    def compute_closed_form_solution_product(self, thisD, y ): 
        ''' compute closed form update of a protocentroid for the product aggregator function. 

        Returns: 
        updated protocentroid (ndarray)
        '''
        n,m = thisD.shape 
        solutionvec = np.zeros(m)
        for j in range(m): 
            den = np.sum( y[:, j]**2)
            if den > 0: 
                solutionvec[j] =  ( np.sum([thisD[h,j] *  y[h,j] for h in range(n)])  ) /  den
            else: 
                solutionvec[j] = 0
        return solutionvec.reshape((1, self.D.shape[1])) 
    

    def compute_closed_form_solution_sum(self, thisD, y ): 
        ''' compute closed form update of a protocentroid for the sum aggregator function. 

        Returns: 
        updated protocentroid (ndarray)
        '''
        n,m = thisD.shape 
        solutionvec = np.zeros(m)
        for j in range(m): 
             solutionvec[j] = ( np.sum([thisD[h,j] - y[h,j] for h in range(n)])  ) /  n 
        return solutionvec.reshape((1, self.D.shape[1])) 

    def compute_closed_form_solution(self, thisD, y): 
        ''' compute closed form update of protocentroids for the either the product or the sum aggregator function. 

        Returns: 
        updated protocentroid (ndarray)
        '''
        if self.operator == "product": 
            sol = self.compute_closed_form_solution_product( thisD, y )
        else: 
            sol = self.compute_closed_form_solution_sum( thisD, y ) 
        return sol  
    
    
    
    def update_B1(self, indices, other_B_indices, current_other_B):
        ''' update the first set of protocentroids 

        Returns: 
        updated first set of protocentroids (ndarray)
        '''
        B = np.zeros((self.r1, self.m)) 
        for i in range(self.r1): 
            idxs = np.where(indices == i)[0] # data points belonging to cluster i 
            # if at least one observation is assigned to each cluster, update, otherwise this is useless centroid
            if len(idxs) > 0: 
                all_other_indices =  other_B_indices[idxs] #
                B[i,:] = self.compute_closed_form_solution(self.D[idxs,:],current_other_B[all_other_indices,:])
            else: 
                # get the selected row
                B[i,:]  = self.D[np.random.choice(self.n)]
             
            self.B_1 = B 
            l = compute_loss(self.A_1, self.B_1, self.A_2, self.B_2, self.D, self.operator) 
        return B 
        
    
    
    def update_B2(self, indices, other_B_indices, current_other_B):
        ''' update the second set of protocentroids 

        Returns: 
        updated first set of protocentroids (ndarray)
        '''
        B = np.zeros((self.r2, self.m)) 
        for i in range(self.r2): 
            idxs = np.where(indices == i)[0] # data points belonging to cluster i 
            # if at least one observation is assigned to each cluster, update, otherwise this is useless centroid
            if len(idxs) > 0: 
                all_other_indices =  other_B_indices[idxs] #
                B[i,:] = self.compute_closed_form_solution(self.D[idxs,:],current_other_B[all_other_indices,:])
            else: 
                B[i,:]  = self.D[np.random.choice(self.n)]
             
            self.B_2 = B 
            l = compute_loss(self.A_1, self.B_1, self.A_2, self.B_2, self.D, self.operator) 
        
        return B 

        
        
    
    
    def fit(self, n_iter, th_movement = 0.0001,  verbose = False, init_type = "random"): 
        ''' 
        Perform Khatri-Rao K-Means clustering

        Returns: 
        A_1: first assignment matrix (ndarray) 
        B_1: first set of protocentroids (ndarray)  
        A_2: second assignment matrix (ndarray) 
        B_2: second set of protocentroids (ndarray) 
        l: inertia of the solution (float) 
        idxs_all: cluster membership (i.e., centroid assingments) 
        '''
        # initializitation phase  
        if init_type == "kmeans++ random": 
            B_1 , B_2 = self.kmeans_plus_plus_init_random() 
        elif init_type == "kmeans++ deterministic": 
            B_1 , B_2 = self.kmeans_plus_plus_init_deterministic() 
        elif init_type == "random": 
            B_1 , B_2 = self.random_inizialization_B() 
        elif init_type == "ones": 
            B_1, B_2 = self.init_ones()
        elif isinstance(init_type, list):
            B_1, B_2 = init_type    
        else:
            print("Invalid input. Please input a list of semi-centroids or enter a valid string specifying the initialization type (kmeans++ random / kmeans++ deterministic / random). Defaulting to random.")
            B_1 , B_2 = self.random_inizialization_B() 
    
        
        self.allcenters = np.full((self.r1 * self.r2, self.m), np.inf, dtype="float32")
        
        for itr in range(n_iter): 
            
            # update centroids 
            self.previous_allcenters = np.copy(self.allcenters) 
            all_centers = find_current_centers(B_1,B_2, self.r1, self.r2, self.m, self.operator)  
            self.allcenters = all_centers

            # update assignments 
            indices_B_1, indices_B_2, idxs_all = self.find_cluster_membership(self.D) 
            A_1 = self.update_A(indices_B_1, self.r1) 
            A_2 = self.update_A(indices_B_2, self.r2) 
            
            self.A_1 = A_1 
            self.A_2 = A_2 
            self.B_1 = B_1 
            self.B_2 = B_2 
            
            # update protocentroids 
            B_1 = self.update_B1(indices_B_1, indices_B_2, B_2) 
            B_2 = self.update_B2(indices_B_2, indices_B_1, B_1)
            
            if verbose==2: 
                l = compute_loss(A_1, B_1, A_2, B_2, self.D, self.operator) 
                print("iteration: " + str(itr) + " \\ loss: " + str(l))
             
            # check convergence 
            movement_centroids = compute_centroid_movement(self.allcenters, self.previous_allcenters) 
            
            if  movement_centroids < th_movement: 
                print(f"Converged after {itr + 1} iterations.")
                break
                        
        if itr == n_iter-1: 
            print(f"Maximum number of iterations ({itr + 1}) reached.")

        # compute solution inertia 
        l = compute_loss(A_1, B_1, A_2, B_2, self.D, self.operator)
        return  [A_1, B_1, A_2, B_2], l, idxs_all
    
    
    def predict(self, new_data): 
        '''
        Predict cluster membership of unseen data 

        Returns: 
        predicted cluster membership (ndarray)
        '''
        _, _, indices_best = self.find_cluster_membership(new_data)
        return indices_best.flatten()
        
    
    
