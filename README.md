
## Intro
This repo contains the scripts to generate shadow of a car using image processing techniques and predefined logic for images taken in
`front-right`, `front-left`, and `rear-right`.

## Description
This method uses the regular bounding box and the minimum area bounding box to locate points of interest. These points are used to
draw a polygon which serves to be the shadow of the car. These points are determined differently according to their angles. To adjust
the points of interest, modify the `get_polygon` function in `generate_shadow.py`

```python
def get_polygon(min_bbox: list, bbox: list, extreme_pts: list, angle: str):
...

    else:
        if angle == 'front-left':
            mtopleft, mtopright, mbottomright, mbottomleft = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]

            left_slope, left_slope_c = get_slope(mbottomleft, mtopleft)
            left_xinterset = (bottom_horizontal-left_slope_c) /left_slope
            polygon = [l_m, [int(left_xinterset),bottom_horizontal], b_m, r_m] # modify polygon coordinates here
            
...
    return polygon

```

## Set Up
The input of the script `generate_shadow.py` requires an inverse mask image of the car and the car image with transparent background.

The result of the car with synthetic shadow is saved as `res.png`. To run the script:

```python
python generate_shadow.py --angle {angle of car} --foreground {path of car image with transparent background} --mask {path of image mask}
```

