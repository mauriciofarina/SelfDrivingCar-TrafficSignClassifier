
    
    
    cv2.imshow('frame',xTrain[i,:,:,2])
    while(1):
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()