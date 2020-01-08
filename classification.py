#%%k nearest neigbhors 



def knn(xyz,k):
    from sklearn.neighbors import NearestNeighbors #import NearestNeigbhors package
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(xyz) #['auto', 'ball_tree', 'kd_tree', 'brute']
    distances, indices = nbrs.kneighbors(xyz) #the indices of the nearest neighbors
    return indices



#%%principal component analysis 
#https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    

def PCA(xyz, indices):
    import pptk
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import decomposition, datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    linearity = []
    planarity = []
    scatter = []
    omnivariance = []
    anisotropy = []
    #eigenentropy = []
    change_curvature = []
    dif_elev = []
    sum_eigenvalues = []
    mean_elev = []
    omnivariance = []
    
    for i in range(len(indices)):
        ind = indices[i]
        coords = xyz[(ind),:]
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
            #should i extract the mean?
    # =============================================================================
    #     data_x = x - x.mean()
    #     data_y = y - y.mean()
    #     data_z = z - z.mean()
    # =============================================================================
        #xyz_new = np.vstack((data_x,data_y,data_z)) #with normalize data
        xyz_new = np.vstack((x,y,z)) #without normalize data
        cov_matrix = np.cov(xyz_new)
        e ,v = np.linalg.eig(cov_matrix)
        e_sorted = np.sort(e)
        e = e_sorted[::-1]
            omni = (e[0]*e[1]*e[2])**(1/3)
        omnivariance.append(omni)
        lin = (e[0]-e[1])/e[0]
        linearity.append(lin)
        plan = (e[1]-e[2])/e[0]
        planarity.append(plan)
        sc = e[2]/e[0]
        scatter.append(sc)
        anis = (e[0]-e[2])/e[0]
        anisotropy.append(anis)
        #ei = -(e[0]*math.log(e[0])+e[1]*math.log(e[1])+e[2]*math.log(e[2]))
        #eigenentropy.append(ei)
        sum_e = sum(e)
        sum_eigenvalues.append(sum_e)
        cha = e[2]/sum_e
        change_curvature.append(cha)
        m_el = z.mean()
        mean_elev.append(m_el)
        d_el = z.max()-z.min()
        dif_elev.append(d_el)
        
        #normalization
    omnivariance = np.asarray(omnivariance)
    omn_n = (omnivariance -omnivariance.min()) / (omnivariance.max() - omnivariance.min())
    l = np.asarray(linearity)
    lin_n = (l -l.min()) / (l.max() - l.min())
    p = np.asarray(planarity)
    plan_n = (p -p.min()) / (p.max() - p.min())
    s = np.asarray(scatter)
    scat_n = (s -s.min()) / (s.max() - s.min())
    an = np.asarray(anisotropy)
    an_n = (an -an.min()) / (an.max() - an.min())
    #eig = np.asarray(eigenentropy)
    #eig_n = (eig -eig.min()) / (eig.max() - eig.min())
    ch = np.asarray(change_curvature)
    ch_cur_n = (ch -ch.min()) / (ch.max() - ch.min())
    m_e = np.asarray(mean_elev)
    mean_el_n = (m_e -m_e.min()) / (m_e.max() - m_e.min())
    d_e = np.asarray(dif_elev)
    dif_elev_n = (d_e -d_e.min()) / (d_e.max() - d_e.min())
    
    
    
    #visualization
    v = pptk.viewer(xyz,lin_n)
    v.set(point_size=7)
    
    v = pptk.viewer(xyz,plan_n)
    v.set(point_size=7)
    
    v = pptk.viewer(xyz,scat_n)
    v.set(point_size=7)
    
    v = pptk.viewer(xyz,an_n)
    v.set(point_size=7)
    
    #v = pptk.viewer(xyz,eig_n)
    #v.set(point_size=10)
    
    v = pptk.viewer(xyz,ch_cur_n)
    v.set(point_size=7)
    
    v = pptk.viewer(xyz,mean_el_n)
    v.set(point_size=1)
    
    v = pptk.viewer(xyz,dif_elev_n)
    v.set(point_size=7)
    
    v = pptk.viewer(xyz,omn_n)
    v.set(point_size=7)
    
    return None