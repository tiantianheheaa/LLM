## 一句话定义

**TensorRT 是 NVIDIA 提供的高性能深度学习推理引擎 / SDK / C++ 库**，面向生产环境部署，主要用于把已经训练好的神经网络模型优化并运行在 NVIDIA GPU 上，从而获得**低延迟、高吞吐、较低显存占用**的推理效果。它只用于 **Inference 推理**，不用于模型训练。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔4〕](https://m.blog.csdn.net/like_study_cat/article/details/108928576)

---

## 1. TensorRT 的定位：它解决什么问题？

深度学习通常分为两个阶段：**训练 Training** 和 **推理 Inference**。训练阶段通过前向传播和反向传播不断更新权重；推理阶段则使用已经训练好的固定权重，对新输入做预测，只需要前向传播。TensorRT 专门服务于第二个阶段，也就是把训练好的模型高效部署到 GPU 上进行推理。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔5〕](https://m.blog.csdn.net/weixin_45277161/article/details/136566919)

它的核心目标是：在生产环境中部署深度学习应用时，通过图优化、层融合、精度优化、内核选择、内存优化等手段，让模型推理更快、更稳定、更省资源。典型收益包括：**提升吞吐量、降低延迟、减少 GPU 内存占用**。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔4〕](https://m.blog.csdn.net/like_study_cat/article/details/108928576)

---

## 2. TensorRT 的主要作用

### 2.1 加速推理

TensorRT 可以将 TensorFlow、PyTorch、Caffe、ONNX 等来源的模型转换为高性能推理引擎，在 NVIDIA GPU 上运行，从而加速图像分类、目标检测、自然语言处理等实时推理任务。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 2.2 降低延迟、提高吞吐

TensorRT 面向生产部署，强调低时延和高吞吐。它会在构建阶段优化网络，并在部署阶段以最小化延迟、最大化吞吐的方式运行优化后的网络。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 2.3 减少冗余计算和显存占用

TensorRT 会消除未使用的输出层、消除无用操作，并优化神经网络的内存使用，从而减少不必要的计算和 GPU 资源消耗。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 2.4 支持低精度推理

TensorRT 支持 FP16、INT8 等低精度推理模式，在保持可接受精度的前提下提升速度、降低延迟。INT8 模式通常需要做精度校准或设置动态范围。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

---

## 3. TensorRT 的核心优化技术

### 3.1 图优化与无用层消除

TensorRT 会对神经网络计算图进行优化，例如消除未使用的输出层、消除等价于 no-op 的无效操作，避免执行无意义的计算。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 3.2 层融合 / 算子融合

TensorRT 会把可以合并的算子融合成更少的计算步骤，例如将 **Convolution + Bias + ReLU** 融合成一个层，减少中间数据读写和 kernel 调用开销。搜索结果中还提到垂直层融合、水平层融合，以及对相似参数和相同目标张量的操作进行聚合。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 3.3 精度优化

TensorRT 可以使用 FP16、INT8 等混合精度或低精度方式进行推理，以提升速度和降低内存占用。INT8 模式通常需要校准数据集、动态范围设置或量化感知训练等方式辅助确定量化范围。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 3.4 内核自动选择与平台优化

TensorRT 会根据层参数和实际测量性能，从高度优化的内核集合中选择更快的实现，同时执行特定平台优化，生成适合目标 NVIDIA GPU 的推理引擎。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔4〕](https://m.blog.csdn.net/like_study_cat/article/details/108928576)

### 3.5 内存与执行优化

TensorRT 会优化神经网络的内存使用，并支持同步、异步执行、执行上下文、绑定输入输出等运行时机制，以便在服务化或批处理场景中高效推理。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

---

## 4. TensorRT 的组成部分

### 4.1 推理优化器 + 运行时

TensorRT SDK 包含**深度学习推理优化器**和**运行时环境**。优化器负责把训练好的模型转换、优化为高效推理引擎；运行时负责在部署阶段加载引擎并执行推理。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔5〕](https://m.blog.csdn.net/weixin_45277161/article/details/136566919)

### 4.2 C++ 核心库与 Python API

TensorRT 的核心是一个 **C++ 库**，用于在 NVIDIA GPU 上进行高性能推理。同时它也提供 Python API，方便开发者进行模型导入、优化、推理以及前后处理集成。搜索结果提到 TensorRT 在所有支持平台提供 C++ 实现，并在 x86、aarch64、ppc64le 平台上提供 Python 支持。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 4.3 Parser：模型解析器

TensorRT 提供多种解析器，用于导入不同格式的模型，例如 Caffe 解析器、UFF 解析器、ONNX 解析器。模型可以从 Caffe 直接导入，也可以通过 UFF / ONNX 等格式从其他框架导入。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 4.4 Builder：构建器

Builder 根据网络定义和构建配置创建优化后的推理引擎。构建阶段会执行层融合、精度选择、内核搜索、平台优化等过程，最终生成用于部署的 engine / plan。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 4.5 Network Definition：网络定义

网络定义接口用于描述模型结构。开发者可以通过解析器导入已有模型，也可以通过 API 编程式地创建网络、设置层参数和权重。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 4.6 Builder Config：构建配置

Builder 配置用于指定创建引擎时的详细参数，例如最大工作空间、优化配置文件、精度模式、INT8 校准器、自动调优迭代次数等。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 4.7 Engine / Plan：推理引擎

TensorRT 在构建阶段会生成优化后的推理引擎，也常被序列化为 plan 文件。部署阶段可以反序列化该文件并快速创建运行时对象，用于执行推理。需要注意，搜索结果提到生成的 plan 文件不能跨平台或跨 TensorRT 版本移植。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 4.8 Execution Context：执行上下文

执行上下文用于实际运行推理。应用程序通常会创建 execution context，把输入数据拷贝到 GPU，调用异步或同步执行接口，再把输出结果从 GPU 拷回主机侧。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

---

## 5. TensorRT 的工作流程

### 5.1 总体流程

TensorRT 的使用通常分为两个阶段：**构建 build** 和 **部署 deployment**。构建阶段对模型网络进行优化并生成 engine / plan；部署阶段在服务或应用中加载该 engine / plan，接收输入数据，执行推理并返回输出结果。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 5.2 常见使用步骤

1. **准备模型**：先使用 PyTorch、TensorFlow、Caffe 等框架训练模型，并导出为 TensorRT 支持的格式，常见方式是导出为 ONNX。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
2. **导入模型**：使用 TensorRT 的 Parser 解析模型，例如 ONNX Parser、Caffe Parser、UFF Parser。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)  
3. **构建网络定义**：创建 TensorRT builder 和 network，通过解析器填充网络结构。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
4. **配置优化参数**：设置 FP16、INT8、workspace、动态 shape profile、校准器等构建参数。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)  
5. **生成推理引擎**：调用 builder 构建优化后的 CUDA engine，并可序列化为 plan 文件。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
6. **部署运行**：在应用程序中反序列化 engine，创建 execution context，分配 GPU/CPU 缓冲区，拷贝输入，执行推理，读取输出。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  

---

## 6. 典型模型转换路径

如果模型来自 PyTorch，常见路径是：**PyTorch 模型 → ONNX 模型 → TensorRT Engine**。ONNX 作为开放神经网络交换格式，可以帮助不同训练框架的模型转换为统一中间表示，再进一步转换为目标平台支持的格式。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

如果模型来自 Caffe，TensorRT 可以使用 Caffe Parser 读取网络结构文件和权重文件。搜索结果中提到部署分类网络时可能需要网络结构文件 `deploy.prototxt`、训练权重 `net.caffemodel`，以及标签文件、batch size 和输出层定义。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)

---

## 7. TensorRT 如何使用：示意流程

### 7.1 Python API 基本思路

使用 Python API 时，通常先导入 TensorRT，创建 Logger、Builder、Network 和 Parser，解析模型并构建 engine；推理时创建 execution context，把输入数据拷贝到 GPU，调用 `execute_async_v2` 等接口执行推理，再把输出从 GPU 拷回。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 7.2 C++ API 使用思路

C++ API 通常用于性能关键或安全性要求较高的场景。搜索结果提到，C++ API 和 Python API 在能力上接近；Python API 的优势是便于结合 Python 库做数据预处理和后处理，而性能关键场景更适合使用 C++。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

### 7.3 安装与环境要求

TensorRT 依赖 NVIDIA GPU、GPU 驱动、CUDA，通常还需要匹配 cuDNN 等依赖。安装时需要关注操作系统、CUDA 版本、cuDNN 版本、TensorRT 版本之间的兼容性；安装完成后可以通过示例程序或自己的模型验证是否可正常推理。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

---

## 8. TensorRT 的适用场景

### 8.1 计算机视觉

TensorRT 常用于图像分类、目标检测、图像分割、人脸识别、视频分析等视觉任务，这些任务通常对实时性和吞吐量有较高要求。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 8.2 自动驾驶与机器人

自动驾驶、机器人等场景需要实时处理大量传感器数据，例如目标检测、车道线检测、行人识别等，TensorRT 可用于降低推理延迟、提高实时处理能力。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 8.3 自然语言处理与语音

TensorRT 可用于自然语言处理、文本分类、命名实体识别、机器翻译、语音识别、语音合成等任务，也适合云端 AI 服务中的实时推理。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 8.4 推荐系统与云端 AI 服务

搜索结果中提到 TensorRT 可用于推荐、个性化服务、云计算 AI 服务等场景，帮助服务端以更低成本提供高吞吐、低延迟的智能服务。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

### 8.5 边缘计算与嵌入式设备

TensorRT 适用于边缘设备、嵌入式系统、物联网设备等资源受限环境，通过低精度推理、内存优化、模型优化等方式提高部署效率。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

---

## 9. TensorRT 与训练框架的关系

TensorRT 不是 TensorFlow、PyTorch 这类训练框架的替代品，而是与它们互补：模型训练仍然可以在 PyTorch、TensorFlow、Caffe、MXNet 等框架中完成，训练完成后再导出模型，由 TensorRT 进行优化和部署推理。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)

换句话说，训练框架负责“**把模型训练好**”，TensorRT 负责“**把训练好的模型跑得更快**”。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)

---

## 10. 使用 TensorRT 时需要注意的问题

1. **只做推理，不做训练**：TensorRT 不能用来训练模型，只用于训练后模型的推理部署。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)  
2. **依赖 NVIDIA GPU 和 CUDA**：TensorRT 依赖 CUDA 进行 GPU 计算，面向 NVIDIA GPU 设计。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
3. **版本兼容很重要**：需要关注 TensorRT、CUDA、cuDNN、GPU 驱动、操作系统、Python 版本之间的匹配。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)  
4. **Plan / Engine 通常不具备跨平台可移植性**：搜索结果提到生成的 plan 文件不能跨平台或跨 TensorRT 版本移植，通常应在目标部署环境或兼容环境中构建。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
5. **INT8 需要校准或动态范围设置**：为了获得较好精度，INT8 推理通常需要代表性校准数据集，或者为张量设置动态范围。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  
6. **构建阶段可能耗时较长**：TensorRT 在构建阶段会做优化、内核选择和性能搜索，因此构建引擎可能比较耗时；典型做法是构建一次并序列化，部署时直接加载。[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)  

---

## 11. 总结

TensorRT 可以理解为 NVIDIA GPU 上的**深度学习模型推理加速器**：它接收训练好的模型，通过图优化、层融合、精度优化、内核选择、内存优化等方式生成高效推理引擎，并在部署阶段提供低延迟、高吞吐的推理能力。它特别适合视觉、语音、NLP、推荐、自动驾驶、机器人、云端服务和边缘设备等对实时性和性能要求较高的场景。[〔1〕](https://m.blog.csdn.net/weixin_33786077/article/details/85987131)[〔2〕](https://devpress.csdn.net/v1/article/detail/136566919)[〔3〕](https://m.blog.csdn.net/weixin_41010198/article/details/107604593)
