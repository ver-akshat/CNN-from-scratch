# Implementing a Convolutoinal Neural network from scratch using Python and Numpy only.

import numpy as np

class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size):
        self.kernel_num=kernel_num
        self.kernel_size=kernel_size
        # generating random filters of shape kernel_num, kernel_size, kernel_size
        self.kernels=np.random.randn(kernel_num,kernel_size,kernel_size)/(kernel_size**2)

    def patches_generator(self,image):
        """
        Divide input image into patches to be used during convolution and outputs tuples containing
        pathces and their coordinates.
        """
        # extracting image height and width
        image_h,image_w=image.shape
        self.image=image
        # Acc to CNn theory no of patches, given filter size (kernel Size =f) is h-f+1 , w-f+! for hieght and width resp
        for h in range(image_h-self.kernel_size+1):
            for w in range(image_w-self.kernel_size+1):
                patch=image[h:(h+self.kernel_size),w:(w+self.kernel_size)]
                yield patch,h,w
    
    def forward_propagation(self,image):
        """
        Perform forward propagation for convolutional layer
        """
        image_h,image_w=image.shape
        # initialize convolutional output volume of correct size
        convolution_output=np.zeros((image_h-self.kernel_size+1,image_w-self.kernel_size+1,self.kernel_num))
        for patch,h,w in self.patches_generator(image):
            # perform convoluiton for each patch
            convolution_output[h,w]=np.sum(patch*self.kernels,axis=(1,2))
        return convolution_output

    def backward_propagation(self,dE_dY,alpha):
        """Takes the gradient of loss function wrt the output and compute gradient of loss funciton wrt 
        kernel's weigths
        """
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        return dE_dk


class SoftmaxLayer:
    """Takes volume coming from convolution and pooling layer, flatten it and uses it for next layers"""
    def __init__(self,input_units,output_units):
        #initialize weights & biases
        self.weight=np.random.randn(input_units, output_units)/input_units
        self.bias=np.zeros(output_units)

    def forward_propagation(self,image):
        self.original_shape=image.shape
        image_flattened=image.flatten()
        self.flattened_input=image_flattened
        first_output=np.dot(image_flattened,self.weight)+self.bias
        self.output=first_output
        softmax_output=np.exp(first_output)/np.sum(np.exp(first_output),axis=0)
        return softmax_output

    def backward_propagation(self,dE_dY,alpha):
        for i,gradient in enumerate(dE_dY):
            if gradient==0:
                continue
            transformation_eq=np.exp(self.output)
            S_total=np.sum(transformation_eq)
            # compute gradient wrt output Z
            dY_dZ=-transformation_eq[i]*transformation_eq/(S_total**2)
            dY_dZ[i]=transformation_eq[i]*(S_total-transformation_eq[i])/(S_total**2)
            # compute gradient of output Z wrt weight,bias and input
            dZ_dw=self.flattened_input
            dZ_db=1
            dZ_dX=self.weight
            # loss gradient wrt output
            dE_dZ=gradient*dY_dZ
            # loss gardient wrt weight,bias and input
            dE_dw=dZ_dw[np.newaxis].T@dE_dZ[np.newaxis]
            dE_db=dE_dZ*dZ_db
            dE_dX=dZ_dX@dE_dZ
            # Update parameters
            self.weight-=alpha*dE_dw
            self.bias-=alpha*dE_db
            return dE_dX.reshape(self.original_shape)


class MaxPoolingLayer:
    def __init__(self,kernel_size):
        self.kernel_size=kernel_size

    def patches_generator(self,image):
        """
        Divide the input image in patches to be used during pooling.
        Yields the tuples containing the patches and their coordinates.
        """
        output_h=image.shape[0]//self.kernel_size
        output_w=image.shape[1]//self.kernel_size
        self.image=image
        for h in range(output_h):
            for w in range(output_w):
                patch=image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size),(w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                yield patch,h,w

    def forward_propagation(self,image):
        image_h,image_w,num_kernels=image.shape
        max_pooling_output=np.zeros((image_h//self.kernel_size,image_w//self.kernel_size,num_kernels))
        for patch,h,w in self.patches_generator(image):
            max_pooling_output[h,w]=np.amax(patch,axis=(0,1))
        return max_pooling_output

    def backward_propagation(self,dE_dY):
        """
        Takes the gradient of the loss function wrt the output and computes the gradients of the loss function wrt the kernels'
        weights.dE_dY comes from the following layer, typically softmax.
        There are no weights to update, but the output is needed to update the weights of the convolutional layer.
        """
        dE_dk=np.zeros(self.image.shape)
        for patch,h,w in self.patches_generator(self.image):
            image_h,image_w,num_kernels=patch.shape
            max_val=np.amax(patch,axis=(0,1))
            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h,idx_w,idx_k]==max_val[idx_k]:
                            dE_dk[h*self.kernel_size+idx_h,w*self.kernel_size+idx_w,idx_k]=dE_dk[h,w,idx_k]
            return dE_dk

def CNN_forward(image,label,layers):
    output=image/255.
    for layer in layers:
        output=layer.forward_propagation(output)
    loss=-np.log(output[label])
    accuracy=1 if np.argmax(output) == label else 0
    return output,loss,accuracy

def CNN_backprop(gradient,layers,alpha=0.05):
    grad_back=gradient
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer,SoftmaxLayer]:
            grad_back=layer.backward_propagation(grad_back,alpha)
        elif type(layer)==MaxPoolingLayer:
            grad_back=layer.backward_propagation(grad_back)
    return grad_back

def CNN_training(image,label,layers,alpha=0.05):
    # forward step
    output,loss,accuracy=CNN_forward(image,label,layers)
    # initial gradient
    gradient=np.zeros(10)
    gradient[label]=-1/output[label]
    # backward step
    gradient_back=CNN_backprop(gradient,layers,alpha)
    return loss,accuracy