import numpy as np 

def find_current_centers(B_1, B_2, r1, r2, m, operator):
    """
    Computes the Cartesian product of B_1 and B_2 to generate combined feature vectors (centroids).

    Parameters:
    B_1 (ndarray): A (r1, m) array representing the first set of basis vectors.
    B_2 (ndarray): A (r2, m) array representing the second set of basis vectors.
    r1 (int): Number of rows in B_1.
    r2 (int): Number of rows in B_2.
    m (int): Dimensionality of each feature vector.

    Returns:
    ndarray: An array of shape (r1 * r2, m) where each row is the element-wise product of a row from B_1 and a row from B_2.
    """
    reshaped_B_1 = np.reshape(B_1, (r1, 1, m))
    reshaped_B_2 = np.reshape(B_2, (1, r2, m))
       
    if operator == "product": 
        allcenters = (reshaped_B_1 * reshaped_B_2).reshape(r1*r2,m)
    else: 
        allcenters = (reshaped_B_1 + reshaped_B_2).reshape(r1*r2,m)
    
    return allcenters

def compute_loss(A_1, B_1, A_2, B_2,D, operator):
    """
    Computes the squared reconstruction loss between the model output and target data.

    The output is formed by element-wise multiplication of two matrix products: (A_1 @ B_1) and (A_2 @ B_2).

    Parameters:
    A_1 (ndarray): Left matrix to multiply with B_1.
    B_1 (ndarray): Right matrix to multiply with A_1.
    A_2 (ndarray): Left matrix to multiply with B_2.
    B_2 (ndarray): Right matrix to multiply with A_2.
    D (ndarray): Target data to approximate.

    Returns:
    float: The total squared error between the model prediction and target D.
    """
    if operator == "product":
        loss = np.sum(np.power(((A_1 @ B_1) * (A_2 @ B_2) - D),2))
    else:
        loss = np.sum(np.power(((A_1 @ B_1) + (A_2 @ B_2) - D),2))
    return loss
  
def compute_centroid_movement(allcenters, previous_allcenters): 
    """
    Calculates the total squared movement of centroids between iterations.

    Parameters:
    allcenters (ndarray): Current centroid positions.
    previous_allcenters (ndarray): Centroid positions from the previous iteration.

    Returns:
    float: Sum of squared differences between current and previous centroid positions.
    """
    return np.sum( np.power((allcenters - previous_allcenters),2)  )