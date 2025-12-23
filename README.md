ReadME

# 环境配置

使用 `uv` 进行环境配置，请确保你已经安装了uv

~~~bash
# 使用gpu版本的pytorch
uv sync --extra cu121
# 使用cpu版本的pytorch
uv sync --extra cpu
~~~


# 数据集介绍

[数据集介绍文档](./doc/dataset.md)

# 数据预处理

[数据预处理文档](./doc/data_preprocess.md)

# 模型构建

[模型文档](./doc/model.md)