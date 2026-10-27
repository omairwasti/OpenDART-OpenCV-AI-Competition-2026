function candidates = detect_asteroids_opencv(image)
%DETECT_ASTEROIDS_OPENCV Call Python OpenCV with pixels only.
% Input: grayscale or RGB uint8 matrix. Output centroid_px is [u,v] in
% zero-based OpenCV coordinates. An N-by-M MATLAB uint8 matrix is converted
% explicitly into a NumPy ndarray, avoiding orientation/reshape ambiguity.
if ~isa(image, 'uint8') || ndims(image) > 3
    error('OpenDART:ImageType', 'Pass a uint8 image matrix.');
end
pixels = py.numpy.array(image, pyargs('dtype', py.numpy.uint8, 'order', 'C'));
candidates = jsondecode(char(py.opendart.matlab_bridge.detect_json(pixels)));
end

