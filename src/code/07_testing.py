def attention_map(inpt, channels):
    s_o = inpt[0]  # shape (batch_size, width=25)
    t_o = inpt[1]  # shape (batch_size, height=300)
    ipt = inpt[2]   # shape (batch_size, height=300, width=25, channels=3)

    height = 300
    width = 25
    
    ''' adaptive spatial attention '''
    s_o = K.l2_normalize(s_o, axis=1)
    s_map = K.expand_dims(s_o, axis=1)  # shape (batch_size, 1, width)
    s_o = K.expand_dims(s_o, axis=1)
    for h in range(height-1):
        s_map = K.concatenate([s_map, s_o], axis=1)  # shape (batch_size, height, width)

    ''' frame-level temporal attention '''
    t_o = K.l2_normalize(t_o, axis=1)
    t_map = K.expand_dims(t_o, axis=2)  # shape (batch_size, height, 1)
    t_o = K.expand_dims(t_o, axis=2)
    for w in range(width-1):
        t_map = K.concatenate([t_map, t_o], axis=2)  # shape (batch_size, height, width)

    ''' Prior Spatial Attention '''
    a_o = multiply([s_map, t_map])  # shape (batch_size, height, width)
    
    # Create ROI mask (now without batch dimension)
    roi_mask = np.zeros((300, 25))  # shape (height, width)
    for i in range(300):
        for j in [3,8,12,14,17,19]:
            roi_mask[i, j] = 1
    
    # Convert to tensor and let broadcasting handle batch dimension
    roi_mask = tf.constant(roi_mask, dtype=tf.float32)  # shape (300,25)
    a_o = a_o + roi_mask  # Automatically broadcasts to (batch_size,300,25)
    
    ''' Expand for channels '''
    a_map = K.expand_dims(a_o, axis=3)  # shape (batch_size, height, width, 1)
    a_o = K.expand_dims(a_o, axis=3)
    for c in range(channels-1):
        a_map = K.concatenate([a_map, a_o], axis=3)  # shape (batch_size, height, width, channels)
    
    out = multiply([a_map, ipt])  # element-wise multiply with input
    return out


model = load_model("/kaggle/working/model/df_ytb_Meso2.h5", custom_objects={'X_plus_Layer': X_plus_Layer, 'attention_map': attention_map})
loss, accuracy = model.evaluate([x_test_mit, x_test_meso], y_test, batch_size=32, verbose=1)
print(f"Test Accuracy: {accuracy}")
print(f"Test Loss: {loss:.4f}")