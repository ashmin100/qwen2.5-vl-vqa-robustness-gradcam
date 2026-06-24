# Phase 2 Semantic Perturbation + Clean Table

This table isolates the clean image baseline and the 13 semantic perturbation image-file evaluations used as the Phase 3 Grad-CAM input set.

| no | condition | group | decision | quality | output_tokens | latency_sec | decision_flip | safety_object_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clean | clean | Do not proceed | 5 | 27 | 3.6 | False | [] |
| 2 | weather_fog_mild | weather | Do not proceed | 5 | 50 | 4.9 | False | [] |
| 3 | weather_fog_dense | weather | Do not proceed | 4 | 34 | 3.99 | False | [] |
| 4 | weather_rain_streaks | weather | Do not proceed | 5 | 34 | 3.99 | False | [] |
| 5 | weather_snow_particles | weather | Do not proceed | 5 | 34 | 3.95 | False | [] |
| 6 | weather_dust_haze | weather | Do not proceed | 5 | 36 | 4.09 | False | [] |
| 7 | illumination_night_low_light | illumination | Cannot determine | 5 | 64 | 5.62 | False | ['pedestrian crossing'] |
| 8 | illumination_sun_glare | illumination | Do not proceed | 5 | 36 | 4.07 | False | [] |
| 9 | camera_motion_blur | camera | Do not proceed | 4 | 29 | 3.73 | False | ['pedestrian crossing'] |
| 10 | camera_defocus_blur | camera | Cannot determine | 4 | 28 | 3.61 | False | ['pedestrian crossing'] |
| 11 | camera_low_light_sensor_noise | camera | Cannot determine | 5 | 64 | 5.63 | False | ['pedestrian crossing'] |
| 12 | camera_windshield_droplets | camera | Do not proceed | 5 | 27 | 3.56 | False | [] |
| 13 | camera_jpeg_q45 | camera | Do not proceed | 5 | 32 | 3.66 | False | [] |
| 14 | camera_resolution_drop_070 | camera | Do not proceed | 6 | 37 | 4.0 | False | [] |
