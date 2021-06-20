# Weak Multi-View Supervision for Surface Mapping Estimation
Datasets and dataloader code for [Weak Multi-View Supervision for Surface Mapping Estimation](https://arxiv.org/abs/2105.01388) in PyTorch

> We propose a weakly-supervised multi-view learning approach to learn category-specific surface mapping without dense annotations. We learn the underlying surface geometry of common categories, such as human faces, cars, and airplanes, given instances from those categories. While traditional approaches solve this problem using extensive supervision in the form of pixel-level annotations, we take advantage of the fact that pixel-level UV and mesh predictions can be combined with 3D reprojections to form consistency cycles. As a result of exploiting these cycles, we can establish a dense correspondence mapping between image pixels and the mesh acting as a self-supervisory signal, which in turn helps improve our overall estimates. Our approach leverages information from multiple views of the object to establish additional consistency cycles, thus improving surface mapping understanding without the need for explicit annotations. We also propose the use of deformation fields for predictions of an instance specific mesh. Given the lack of datasets providing multiple images of similar object instances from different viewpoints, we generate and release a multi-view ShapeNet Cars and Airplanes dataset created by rendering ShapeNet meshes using a 360 degree camera trajectory around the mesh. For the human faces category, we process and adapt an existing dataset to a multi-view setup. Through experimental evaluations, we show that, at test time, our method can generate accurate variations away from the mean shape, is multi-view consistent, and performs comparably to fully supervised approaches.

# Dataset

![Multi View Dataset for multiple categories](https://user-images.githubusercontent.com/7645118/122668270-14c41500-d16c-11eb-85e2-3de7956634a8.png)

![Numerous annotations for each instance](https://user-images.githubusercontent.com/7645118/122668273-168dd880-d16c-11eb-84fa-fefa7685b32e.png)

# Annotations Provided

![Annotations Provided](https://user-images.githubusercontent.com/7645118/122668245-feb65480-d16b-11eb-8f6a-2975e7b086d8.png)

# Dataset Download Links

* [Shapenet Cars](https://cdn.fyusion.com/0/ML/shapenet_cars.tar)
* [Shapenet Planes](https://cdn.fyusion.com/0/ML/shapenet_planes.tar)
* 300W_LP (faces) [[Part 1](https://cdn.fyusion.com/0/ML/300W_LP.tar.gz.partaa)], [[Part 2](https://cdn.fyusion.com/0/ML/300W_LP.tar.gz.partab)], [[Part 3](https://cdn.fyusion.com/0/ML/300W_LP.tar.gz.partac)], [[Part 4](https://cdn.fyusion.com/0/ML/300W_LP.tar.gz.partad)]

# Credits
In case the dataset is useful, please cite our work using,
~~~
@inproceedings{rai2021weak,
  title={Weak Multi-View Supervision for Surface Mapping Estimation},
  author={Rai, Nishant and Liaudanskas, Aidas and Rao, Srinivas and Cayon, Rodrigo Ortiz and Munaro, Matteo and Holzer, Stefan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2759--2768},
  year={2021}
}
~~~
