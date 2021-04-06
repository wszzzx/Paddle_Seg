import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import  to_variable

class BasicModel(fluid.dygraph.Layer):
    def __init__(self,nums_classes = 59):
        super(BasicModel,self).__init__()
        self.pool = fluid.Pool2D(pool_size=2,pool_stride=2)
        self.conv = fluid.Conv2D(num_channels=3,num_filters=1,filter_size=1)
    def forward(self,inputs):
        x = self.pool(inputs)
        x = paddle.nn.functional.interpolate(x,size= (inputs.shape[2],inputs.shape[3]))
        return x


def main():
    place = paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = BasicModel(nums_classes=59)
        model.eval()
        input_data = np.random.rand(1,3,8,8).astype(np.float32)
        input_data = to_variable(input_data)
        output_data = model(input_data)
        print(output_data)
        output_data = output_data.numpy()
        print(output_data.shape)
if __name__=="__main__":
    main()