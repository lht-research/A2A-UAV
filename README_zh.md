# A2A-UAV

[English](README.md) · [数据卡](docs/DATASET_CARD.md)

A2A-UAV 是面向空对空无人机检测和识别的多模态数据集。发布内容包括 RGB、全景、热红外、射频时频图、可用的雷达帧和样本元数据。

## 数据获取

预览包 `A2A-UAV_preview.tar.gz` 可通过以下任一入口下载。

- [Google Drive](https://drive.google.com/file/d/1oCtjoF5vfH3MsPDNQEOQ84MvRywHSUso/view?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1c7TXhvHKnL0h4Wx4UmDwog?pwd=A2AD)（提取码 `A2AD`）

完整数据集将在论文录用后公开。

完整发布包由 271 个可独立解压的归档组成，总计约 2.50 TB。数据公开后，将归档解压到同一个空目录，即可得到 `UAV_full_dataset_release/`。

数据共 105,294 个样本，覆盖 15 个无人机标签。训练、验证和测试集分别为 61,767、16,468、27,059 个样本。公开数据省略了采集时间戳。记录标识用于分组，样本编号不表示时间顺序。雷达帧偏移以其对应样本的参考时刻为基准。

## 代码

- [样本读取与归档校验](README.md#dataset)
- [标注转换、全景切片与检测评估](README.md#detection)
- [RF 索引、输入准备与识别实验](README.md#rf-identification)

引用和许可见 [CITATION.cff](CITATION.cff) 和 [LICENSE.md](LICENSE.md)。
