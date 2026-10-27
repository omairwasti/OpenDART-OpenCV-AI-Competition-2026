function selftest_opendart_python_bridge(pythonExecutable)
%SELFTEST_OPENDART_PYTHON_BRIDGE Check MATLAB/Python image coordinates and types.
% This test deliberately uses a non-square image and off-center objects.
if nargin<1, pythonExecutable=''; end
setup_opendart_python(pythonExecutable,true);
[U,V]=meshgrid(0:508,0:316);
image=zeros(317,509,'uint8');
image((U-129).^2+(V-89).^2<=23^2)=180;
image((U-356).^2+(V-219).^2<=10^2)=220;
c=detect_asteroids_opencv(image);
assert(numel(c)>=2,'OpenDART:BridgeDetections','Expected two bright disks.');
centers=reshape([c.centroid_px],2,[]).';
assert(min(vecnorm(centers-[129,89],2,2))<1,'Horizontal/vertical coordinate mismatch.');
assert(min(vecnorm(centers-[356,219],2,2))<1,'Image dimensions were transposed.');
rgb=cat(3,image,uint8(zeros(size(image))),uint8(zeros(size(image))));
rgbCandidates=detect_asteroids_opencv(rgb);
assert(numel(rgbCandidates)>=2,'RGB->BGR conversion did not preserve detections.');
fprintf('MATLAB/OpenCV pixel bridge self-test passed.\n');
end
