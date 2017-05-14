import hw4_pca as hw4
import os
import sys


pca = hw4.PCA(sys.argv[1],10,10)
#calculate average face
pca.find_mean()
pca.plot_average_face()
centered_data = pca.center_data()
original_data = pca.uncenter_data(centered_data)

pca.save_image(original_data,100,'hw4_p1_original_img.jpg')

centered_data_t = centered_data.T
u,s,v = pca.SVD(centered_data_t)


numDim = 9
s_red = pca.reduce_dimension(numDim)
eigenFace = pca.find_eigenface(u,s_red,numDim)
pca.save_image(eigenFace,numDim,"hw4_p1_eigenfaces9.jpg")

numDim = 5
s_red = pca.reduce_dimension(numDim)
eigenFace = pca.find_eigenface(u,s_red,numDim)
#pca.save_image(eigenFace,numDim,"hw4_p1_eigenfaces9.jpg")

#Plot the eigen face
#pca.save_image(eigenFace,numDim,"hw4_p1_eigenfaces5.jpg")
recon = pca.reconstruct_data(u,s_red,v)

#Plot the reconstruct face
pca.save_image(recon,100,"hw4_p1_reconstructed.jpg")
pca.plot_error()
print('completed!')
