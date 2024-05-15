# Circle-Vanishing-Point-Detection
This is a project to detect vanishing points in an image taken inside a pipes.

# Requirements
| list  | Version |
| ------------- | ------------- |
| python  | 3.11.7  |
| numpy | 1.24.1  |
| pandas | 2.1.4  |
| opencv-python | 4.9.0.80  |
| pillow | 9.3.0  |
| skimage | 0.22.0  |

# usage
Duplicate and import the basics
```ruby
import CBVP
from CBVP import cv2, np, pd, plt, Image
```

After loading the image, calculate the Hough transform and vanishing point coordinates
```ruby
path = 'dataset/PVC/b/'
img_array = np.array(Image.open(path+'label.png'))
img_array[img_array != 0] = 255
VP_Detection = CBVP.Vanishing_Point(image_path=path+'img.png',
                             dist=cam_dist, mtx=cam_mtx)
VP_Detection.master_bin= 100
VP_Detection.mask= img_array
VP_Detection.image_processing()
slope_dR_DF, slopes, dRs = VP_Detection.slope_dR_comb()
v_F_dF = VP_Detection.v_p_add(data=slope_dR_DF)
```

To see the scatter of vanishing point coordinates
```ruby
VP_Detection.vp_scatter(data=v_F_dF)
```
![20240515_125243](https://github.com/GNU-HydrosystemLAB/Circle-Vanishing-Point-Detection/assets/169818638/bf9c4cbc-b18f-43dc-9abd-3253b3b72ca3)
We apply a filter for vanishing point coordinates with high stochastic frequency and select one vanishing point that is closest to the normalized average value of dR and slope, which are key parameters for vanishing point formation.
```ruby
SdR_F_dF = VP_Detection.vp_filter(data=v_F_dF)
slope_mean, dR_mean, closest_data = VP_Detection.S_dR_nomal(data=SdR_F_dF)
VP_Detection.best_circles_S = closest_data[0]
VP_Detection.best_circles_B = closest_data[1]
VP_Detection.vanishing_point()
```

Visualize the finally selected vanishing point.
```ruby
VP_Detection.drow_circles(VP_Detection.undist_image,
                   [VP_Detection.best_circles_S,VP_Detection.best_circles_B],
                   color=(0,0,255))
VP_Detection.drow_vp(VP_Detection.undist_image,color=(0,0,255))
cv2.imshow('Vanishing Point Detection', VP_Detection.undist_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
