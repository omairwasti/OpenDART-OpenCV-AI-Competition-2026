function candidates = detect_real_image(imagePath, pythonExecutable)
%DETECT_REAL_IMAGE Detect candidates in one archived PNG, with no metric claims.
if nargin < 2, pythonExecutable = ''; end
setup_opendart_python(pythonExecutable, true);
image = imread(imagePath);
if ~isa(image, 'uint8')
    finitePixels = double(image(isfinite(image)));
    lo = min(finitePixels); hi = max(finitePixels);
    image = uint8(255 * max(0, min(1, (double(image)-lo)/max(hi-lo,eps))));
end
candidates = detect_asteroids_opencv(image);
figure('Name','Archived image: detections only');
imageHandle = imagesc(image); %#ok<NASGU>
axis image; colormap gray; hold on;
for k=1:numel(candidates)
    c = candidates(k);
    xy = double(c.centroid_px(:)) + 1; % MATLAB image axes are one-based.
    plot(xy(1), xy(2), 'r+', 'MarkerSize', 12, 'LineWidth', 1.5);
    text(xy(1)+8, xy(2), sprintf('candidate %d, radius %.1f px',k,c.radius_px), ...
        'Color','y');
end
title('Image candidates; one PNG does not establish range or orbital motion');
end
