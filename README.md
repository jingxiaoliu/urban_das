# Urban sensing via seismic source mapping

This is the repository for the following paper. If you use this implementation, please cite our paper:
>* Liu, J., Li, H., Noh, H.Y., Santi P., Biondi, B. & Ratti, C. (2024). Urban Sensing Using Existing Fiber-Optic Networks. [https://arxiv.org/abs/2408.14320](https://arxiv.org/abs/2409.05820)
>* Liu, J., Yuan, S., Dong, Y., Biondi, B., & Noh, H. Y. (2023). TelecomTM: A fine-grained and ubiquitous traffic monitoring system using pre-existing telecommunication fiber-optic cables as sensors. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 7(2), 1-24.
>* Yuan, S., Liu, J., Noh, H. Y., Clapp, R., & Biondi, B. (2024). Using vehicle‐induced DAS signals for near‐surface characterization with high spatiotemporal resolution. Journal of Geophysical Research: Solid Earth, 129(4), e2023JB028033.

## Visualization

The GIFs below showcase the seismic source mapping results from September 20th, using different normalization methods.

- **Map Normalized in Each 10-Minute Window**:
  ![Map normalized in each 10-minute window](media/map.gif)

- **Map Normalized Over One Day**:
  ![Map normalized in one day](media/map_norm.gif)

For a more comprehensive view, a six-day video encompassing the extended data set is available. You can download this video from the [media folder](media/).

## File List

### Data
Contains all data files used and produced by the project.
- **array_pos/**: Stores XML and NPZ files related to array positions.
  - `DASArray_10LT.xml`, `DASArray_10V.xml`: Configuration files for array settings.
  - `cha_save.npz`, `cha_save_LT.npz`: Channel save files.
- **vs/**: Includes NPZ files for various visualization scripts.
  - `const_maps.npz`: Construction maps used in visualization.
  - `map_region1.npz`, `map_region2.npz`, `map_region3.npz`: Region-specific map data.
  - `mask_region1.npz`, `mask_region2.npz`, `mask_region3.npz`: Mask files for each region.
  - `train_maps.npz`, `truck_maps.npz`: Train and truck map data files.

### Media
Stores media files related to the project.
- `video_fast.mp4`: Fast playback version of the map video.
- `video_nonorm_fast.mp4`: Non-normalized fast playback version of the map video.

### Modules
Python modules for handling data and utilities.
- `das_io.py`: Module for input/output operations of DAS data.
- `utils.py`: Utility functions for general operations.

### Scripts
Jupyter notebooks and Python scripts for project setup and execution.
- `script.ipynb`: Script for visualizing seismic source mapping in the three regions.
- `script_construction.ipynb`: Script for mapping a construction site.
- `script_train.ipynb`: Script for mapping a train passing event.
- `script_truck.ipynb`: Script for mapping a truck passing event.

## Contact
Feel free to send any questions to:
- [Jingxiao Liu](mailto:jingxiao@mit.edu)

