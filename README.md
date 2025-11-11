<h><center>Abstract</center></h>





Drowsiness is one of the primary causes of accidents in the transportation, industrial, and at-risk work environments. Over time, as society leans on road-based transportation and long-duration drives, the implications of drowsiness on safety 
have become a serious public safety issue. Studies show that prolonged inattention can significantly delay reaction time, decrease situational awareness, and ultimately relate to poor decision making that threatens the safety of the operator and 
individuals nearby. Identifying drowsiness early can help reduce the severity of accidents, and can help prevent the loss of life and property.  


Historically, drowsiness has been assessed using physiological measurements such as electroencephalogram (EEG), electrocardiogram (ECG), or heart-rate variability (HRV). While it has been shown that these physiological measurements can be accurately used to assess drowsiness, 
they also require the fixation of specific sensors to the individual to monitor. This can become impractical for a .Deep learning has considerably enhanced the reliability of vision-based fatigue detection. Several deep learning models, including Convolutional Neural Networks (CNNs), 
have been extensively invested in for detecting behavioural indicators such as eye-blink frequency and yawning, i.e., due to their aptitude for spatial feature extraction. CNNs rely on localized receptive field sizes somewhat limiting their ability 
to model long-range dependencies in the image. This limitation arises as there may be very subtle behavioural indicators scattered within the regions of a face. Recent advances in transformer-based architectures have transformed the landscape of computer vision tasks. The 


Vision Transformer (ViT), derived from Natural Language Processing (NLP) models, is capable of learning a purely attention-based mechanism that models global relationships among patches of images. By eliminating convolutional layers, 


ViTs are capable of learning contextual representations much better than CNNs, and are more robust to the inherent variation in techniques such as illumination, occlusions, and face orientation. 


Motivated by these benefits, this study implements a multi-class drowsiness detection framework based on the Vision Transformer framework that can classify normal, eye closed, or yawning states. The framework is trained on two openly available face datasets and assessed using multiple 
performance metrics. The aim is to develop a non-contact, real-time, and robust framework for driver assistance systems, surveillance cameras, and industrial monitoring systems. 
