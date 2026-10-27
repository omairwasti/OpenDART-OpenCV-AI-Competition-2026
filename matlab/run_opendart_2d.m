function result = run_opendart_2d(varargin)
%RUN_OPENDART_2D MATLAB-owned planar plant, image EKF, bounded impact guidance.
% No Computer Vision Toolbox is required. Python/OpenCV processes pixels only;
% numerical dynamics, association, state estimation and control run in MATLAB.
% Example:
% run_opendart_2d('PythonExecutable','/path/to/env/bin/python', ...
%     'ShowFigures',true,'MakeVideo',true);
% Physical time is never rescaled. VideoFPS changes playback speed only.
ip = inputParser;
addParameter(ip,'PythonExecutable','');
addParameter(ip,'OutputDirectory',fullfile(fileparts(fileparts(mfilename('fullpath'))),'results','matlab_vision_2d'));
addParameter(ip,'RequireOpenCV5',true,@islogical);
addParameter(ip,'ShowFigures',true,@islogical);
addParameter(ip,'MakeVideo',false,@islogical);
addParameter(ip,'VideoFPS',24,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'FrameStride',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1&&x==floor(x));
addParameter(ip,'Duration',600,@(x)isnumeric(x)&&isscalar(x)&&x>20);
addParameter(ip,'Step',1,@(x)isnumeric(x)&&isscalar(x)&&x>0&&x<=2);
addParameter(ip,'Speed',20,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'UseKnownRadiusRange',false,@islogical);
addParameter(ip,'Seed',7,@(x)isnumeric(x)&&isscalar(x));
parse(ip,varargin{:}); opts = ip.Results;
runtime = setup_opendart_python(opts.PythonExecutable,opts.RequireOpenCV5);
rng(opts.Seed,'twister');
out = char(opts.OutputDirectory); if ~exist(out,'dir'), mkdir(out); end
c = defaults(); c.duration = opts.Duration; c.step = opts.Step;
c.speed = opts.Speed; c.useRange = opts.UseKnownRadiusRange;
[pf,~,~] = binary_state(c.duration,c);
% Nominal approach aims at the scheduled target center. Truth perturbations
% and the onboard prior are defined independently, not copied from truth.
scNominal = pf(:,2) - [0;c.speed*c.duration];
x = [scNominal+[300;0]; 0.15;c.speed+0.02; c.mass0];
[p0,v0,~] = binary_state(0,c);
assumedScPosition = scNominal+[250;20];
assumedScVelocity = [0.10;c.speed];
z = [p0(:,2)-assumedScPosition; v0(:,2)-assumedScVelocity];
P = diag([100^2,100^2,0.2^2,0.2^2]);
t = 0;
nMax = ceil((c.duration+90)/c.step)+2;
% [t, sc_x, sc_y, vx, vy, mass, est_x, est_y, range, est_range,
%  thrust_x, thrust_y, angle_error, accepted, tgo, sigma_position]
history = nan(nMax,16); accepted = 0; rejected = 0; lost = 0;
lastAcceptedTime=-inf; staleHoldTime=0;
deltaV = 0; satTime = 0; status = 'TIMEOUT'; bodyHit = 0;
visible = 'off'; if opts.ShowFigures, visible='on'; end
fig = figure('Visible',visible,'Name','OpenDART MATLAB image feedback', ...
    'Position',[50 50 1150 620],'Color','w');
writer = [];
if opts.MakeVideo
    writer = VideoWriter(fullfile(out,'matlab_vision_2d.avi'),'Motion JPEG AVI');
    writer.FrameRate = opts.VideoFPS; open(writer);
end
writerGuard = onCleanup(@()close_writer(writer));
recordCount=0;
for k=1:nMax
    [p,~,~] = binary_state(t,c);
    theta = atan2(z(2),z(1)); % ideal pointing: attitude is outside this model
    image = render_camera(t,x,theta,c);
    detections = detect_asteroids_opencv(image);
    [selected,assocCost] = associate(detections,z,P,c); %#ok<ASGLU>
    gotMeasurement = false; innovationAngle = NaN;
    if ~isempty(selected)
        [zNew,PNew,ok,innovationAngle] = update_ekf(z,P,selected,theta,c);
        if ok
            z=zNew; P=PNew; accepted=accepted+1; gotMeasurement=true;
            lastAcceptedTime=t;
        else
            rejected=rejected+1;
        end
    else
        lost=lost+1;
    end
    [aDesired,tgo] = guidance(t,z,x(5),c);
    uncertainty = sqrt(max(eig(P(1:2,1:2))));
    stale=(t-lastAcceptedTime)>5;
    if uncertainty > 1500 || tgo <= c.coast || tgo <= 0 || stale
        aDesired=zeros(2,1);
    end
    if stale, staleHoldTime=staleHoldTime+c.step; end
    F = x(5)*aDesired;
    if norm(F)>c.maxForce
        F=F*c.maxForce/norm(F); satTime=satTime+c.step;
    end
    % Constant force cannot consume more propellant than the next interval permits.
    fuelForce=max(x(5)-c.dryMass,0)*c.isp*c.g0/c.step;
    if norm(F)>fuelForce && norm(F)>0, F=F*fuelForce/norm(F); end
    estSc = p(:,2)-z(1:2);
    history(k,:)=[t,x.',estSc.',norm(p(:,2)-x(1:2)),norm(z(1:2)), ...
        F.',innovationAngle,double(gotMeasurement),tgo,uncertainty];
    recordCount=k;
    if mod(k-1,opts.FrameStride)==0
        draw_frame(fig,image,detections,selected,history(1:k,:),t,x,theta,c);
        if ~isempty(writer), writeVideo(writer,getframe(fig)); end
        if opts.ShowFigures, drawnow limitrate; end
    end
    % Both bodies are checked, including after closest approach.
    h = min(c.step,c.duration+90-t);
    if h<=0, break; end
    [xNext,elapsed,hit] = propagate_contact(t,x,F,h,c);
    deltaV = deltaV + c.isp*c.g0*log(x(5)/xNext(5));
    [z,P] = predict_ekf(t,z,P,F,x(5),elapsed,c);
    % Symmetrization removes floating-point asymmetry, not uncertainty.
    P=(P+P.')/2;
    x=xNext; t=t+elapsed;
    if hit>0
        bodyHit=hit;
        if hit==2, status='TARGET_IMPACT'; else, status='PRIMARY_IMPACT'; end
        [pt,~,~]=binary_state(t,c);
        history(k+1,:)=[t,x.',(pt(:,2)-z(1:2)).', ...
            norm(pt(:,2)-x(1:2)),norm(z(1:2)),F.',NaN,0,0, ...
            sqrt(max(eig(P(1:2,1:2))))];
        recordCount=k+1;
        break;
    end
    if t>c.duration && dot(z(1:2),z(3:4))>0
        status='MISS'; break;
    end
end
history=history(1:recordCount,:);
[p,v,~]=binary_state(t,c);
result=struct('status',status,'success',strcmp(status,'TARGET_IMPACT'), ...
    'terminal_time_s',t,'terminal_range_m',norm(p(:,2)-x(1:2)), ...
    'terminal_speed_mps',norm(v(:,2)-x(3:4)), ...
    'collided_body_matlab_index',bodyHit,'delta_v_mps',deltaV, ...
    'propellant_kg',c.mass0-x(5),'accepted_image_measurements',accepted, ...
    'rejected_image_measurements',rejected,'no_candidate_frames',lost, ...
    'saturation_time_s',satTime,'stale_observation_hold_s',staleHoldTime, ...
    'physical_binary_period_s',c.period, ...
    'physical_primary_spin_period_s',c.spin,'known_radius_range_enabled',c.useRange, ...
    'opencv_version',runtime.opencv_version,'python_executable',runtime.python_executable, ...
    'validation_scope','Independent 2D MATLAB synthetic-image demonstration');
names={'time_s','spacecraft_x_m','spacecraft_y_m','vx_mps','vy_mps','mass_kg', ...
    'estimated_x_m','estimated_y_m','true_range_m','estimated_range_m', ...
    'force_x_N','force_y_N','bearing_innovation_rad','accepted', ...
    'estimated_tgo_s','position_sigma_max_m'};
writetable(array2table(history,'VariableNames',names),fullfile(out,'history.csv'));
fid=fopen(fullfile(out,'summary.json'),'w');
fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true)); fclose(fid);
save(fullfile(out,'simulation.mat'),'history','result','c','opts','P','z');
clear writerGuard; % close the movie before announcing completed outputs
draw_frame(fig,image,detections,selected,history,t,x,theta,c);
exportgraphics(fig,fullfile(out,'terminal_view.png'),'Resolution',140);
make_summary(history,c,out,visible);
make_orbit_overview(c,out,visible);
disp(result);
if ~opts.ShowFigures, close(fig); end
end

function c=defaults()
c.period=42900; c.spin=8136; c.separation=1190; c.fraction=0.00925;
c.n=2*pi/c.period; c.GM=c.n^2*c.separation^3;
c.mu=c.GM*[1-c.fraction,c.fraction];
c.radius=[425,87.5]; c.phase=0;
c.mass0=500; c.dryMass=470; c.maxForce=10; c.isp=220; c.g0=9.80665;
c.pixels=512; c.focal=900; c.cx=(c.pixels-1)/2;
c.coast=20; c.qAcceleration=2e-4;
c.useRange=false;
end

function [p,v,a]=binary_state(t,c)
ang=c.phase+c.n*t;
d=c.separation*[cos(ang);sin(ang)];
dd=c.n*[-d(2);d(1)];
p=d*[-c.fraction,1-c.fraction];
v=dd*[-c.fraction,1-c.fraction]; a=-c.n^2*p;
end

function [g,G]=gravity(t,position,c)
[p,~,~]=binary_state(t,c); g=zeros(2,1); G=zeros(2);
for j=1:2
    d=position-p(:,j); r=max(norm(d),1e-6);
    g=g-c.mu(j)*d/r^3;
    G=G+c.mu(j)*(3*(d*d.')/r^5-eye(2)/r^3);
end
end

function dx=plant_rhs(t,x,F,c)
g=gravity(t,x(1:2),c);
dx=[x(3:4);g+F/x(5);-norm(F)/(c.isp*c.g0)];
end

function y=rk4(fun,t,x,h)
a=fun(t,x); b=fun(t+h/2,x+h*a/2);
d=fun(t+h/2,x+h*b/2); e=fun(t+h,x+h*d);
y=x+h*(a+2*b+2*d+e)/6;
end

function s=surface_values(t,x,c)
p=binary_state(t,c);
s=[norm(x(1:2)-p(:,1))-c.radius(1),norm(x(1:2)-p(:,2))-c.radius(2)];
end

function [x,elapsed,hit]=propagate_contact(t,x,F,h,c)
elapsed=0; hit=0;
while elapsed<h-1e-12
    dt=min(0.2,h-elapsed); x0=x; t0=t+elapsed;
    [centers,velocities,~]=binary_state(t0,c);
    for b=1:2
        speedBound=norm(x0(3:4)-velocities(:,b))+1+c.maxForce/x0(5)*dt;
        clearance=norm(x0(1:2)-centers(:,b))-c.radius(b);
        if clearance<=speedBound*dt
            dt=min(dt,0.25*c.radius(b)/speedBound);
        end
    end
    xn=rk4(@(tt,xx)plant_rhs(tt,xx,F,c),t0,x0,dt);
    s0=surface_values(t0,x0,c); s1=surface_values(t0+dt,xn,c);
    bodies=find(s0>0 & s1<=0); earliest=inf; which=0;
    for j=bodies
        lo=0; hi=dt;
        for n=1:36
            mid=(lo+hi)/2; xm=rk4(@(tt,xx)plant_rhs(tt,xx,F,c),t0,x0,mid);
            sm=surface_values(t0+mid,xm,c);
            if sm(j)>0, lo=mid; else, hi=mid; end
        end
        if hi<earliest, earliest=hi; which=j; end
    end
    if which>0
        x=rk4(@(tt,xx)plant_rhs(tt,xx,F,c),t0,x0,earliest);
        elapsed=elapsed+earliest; hit=which; return;
    end
    x=xn; elapsed=elapsed+dt;
end
end

function dz=relative_rhs(t,z,u,c)
[p,~,a]=binary_state(t,c);
g=gravity(t,p(:,2)-z(1:2),c);
dz=[z(3:4);a(:,2)-g-u];
end

function [z,P]=predict_ekf(t,z,P,F,mass,dt,c)
% Continuous-discrete EKF covariance with gravity-gradient state transition.
[p,~,~]=binary_state(t,c); [~,G]=gravity(t,p(:,2)-z(1:2),c);
A=[zeros(2),eye(2);G,zeros(2)]; Phi=expm(A*dt);
q=c.qAcceleration^2;
Q=q*[dt^3/3*eye(2),dt^2/2*eye(2);dt^2/2*eye(2),dt*eye(2)];
flow=norm(F)/(c.isp*c.g0);
z=rk4(@(tt,zz)relative_rhs(tt,zz,F/(mass-flow*(tt-t)),c),t,z,dt);
P=Phi*P*Phi.'+Q;
end

function [candidate,best]=associate(candidates,z,P,c)
candidate=[]; best=inf;
if isempty(candidates), return; end
r=max(norm(z(1:2)),1); predictedRadius=c.focal*c.radius(2)/r;
J=[-z(2),z(1)]/r^2;
varU=max(9,c.focal^2*(J*P(1:2,1:2)*J.'));
for k=1:numel(candidates)
    d=candidates(k);
    if d.touches_border || d.radius_px<1.2, continue; end
    uv=double(d.centroid_px(:)); covU=double(d.covariance_px2(1,1));
    cost=(uv(1)-c.cx)^2/(varU+max(covU,1)) + ...
        (log(max(d.radius_px,0.1)/max(predictedRadius,0.1))/0.50)^2;
    if cost<best && cost<20
        best=cost; candidate=d;
    end
end
end

function [z,P,ok,innov]=update_ekf(z,P,d,theta,c)
r=norm(z(1:2));
uv=double(d.centroid_px(:)); b=atan((uv(1)-c.cx)/c.focal);
predicted=wrap(theta-atan2(z(2),z(1)));
innov=wrap(b-predicted);
H=[z(2)/r^2,-z(1)/r^2,0,0];
R=max(1.0,double(d.covariance_px2(1,1)))/c.focal^2;
if c.useRange
    % Deliberately optional: equivalent bright-area radius is biased by
    % illumination, shape, occlusion and clipping. This is not stereo range.
    measuredRange=c.focal*c.radius(2)/max(d.radius_px,1);
    innov=[innov;measuredRange-r];
    H=[H;z(1)/r,z(2)/r,0,0];
    R=diag([R,(0.30*r)^2]);
end
S=H*P*H.'+R;
nis=innov.'*(S\innov); ok=isfinite(nis) && nis<16;
if ok
    K=P*H.'/S; z=z+K*innov;
    A=eye(4)-K*H; P=A*P*A.'+K*R*K.';
end
innov=innov(1);
end

function [u,tgo]=guidance(t,z,mass,c) %#ok<INUSD>
r=z(1:2); v=z(3:4); tgo=-dot(r,v)/max(dot(v,v),1e-9);
if tgo<=c.coast, u=zeros(2,1); return; end
[p,~,a]=binary_state(t,c); g=gravity(t,p(:,2)-r,c);
drift=a(:,2)-g;
zem=r+v*tgo+0.5*drift*tgo^2;
u=3*zem/tgo^2;
end

function image=render_camera(t,x,theta,c)
% A planar pinhole observation rendered as textured disks. Pixel processing
% receives only this image; centers/depth never leave this renderer.
[p,~,~]=binary_state(t,c);
forward=[cos(theta);sin(theta)]; lateral=[sin(theta);-cos(theta)];
rel=p-x(1:2); depths=forward.'*rel;
[U,V]=meshgrid(0:c.pixels-1,0:c.pixels-1);
image=2+0.8*randn(c.pixels);
[~,order]=sort(depths,'descend'); % nearer objects overwrite farther objects
for j=order
    if depths(j)<=0, continue; end
    center=c.cx+c.focal*(lateral.'*rel(:,j))/depths(j);
    rp=c.focal*c.radius(j)/depths(j);
    if rp<0.7, continue; end
    xx=(U-center)/rp; yy=(V-c.cx)/rp;
    mask=xx.^2+yy.^2<=1;
    if j==1, spin=2*pi*t/c.spin; else, spin=c.n*t; end
    texture=0.88+0.08*sin(8*xx+spin).*cos(9*yy-spin);
    limb=sqrt(max(0,1-xx.^2-yy.^2));
    % Mild asymmetry stresses brightness centroid bias without pretending to
    % reproduce DRACO photometry or the irregular three-dimensional shapes.
    brightness=(55+140*limb).*(0.95+0.05*xx).*texture;
    image(mask)=brightness(mask)+2*randn(nnz(mask),1);
end
image=uint8(max(0,min(255,image)));
end

function draw_frame(fig,image,detections,selected,h,t,x,theta,c)
figure(fig); clf(fig);
subplot(2,2,[1 3]); hold on; axis equal; grid on;
plot(h(:,2),h(:,3),'b-','LineWidth',1.3);
plot(h(:,7),h(:,8),'--','Color',[0.6 0.3 0.7]);
[p,~,~]=binary_state(t,c); th=linspace(0,2*pi,100);
for j=1:2
    plot(p(1,j)+c.radius(j)*cos(th),p(2,j)+c.radius(j)*sin(th),'k-');
end
spin=2*pi*t/c.spin;
plot([p(1,1),p(1,1)+c.radius(1)*cos(spin)], ...
    [p(2,1),p(2,1)+c.radius(1)*sin(spin)],'Color',[.7 .3 .1]);
plot(x(1),x(2),'bo','MarkerFaceColor','b');
plot(p(1,2),p(2,2),'ro','MarkerFaceColor','r');
plot(p(1,2),p(2,2),'r+');
xlabel('Barycentric x (m)'); ylabel('Barycentric y (m)');
title(sprintf('Physical time %.1f s; terminal approach',t));
legend('Spacecraft truth','Image EKF','Bodies','Location','southwest');
subplot(2,2,2); imagesc(0:c.pixels-1,0:c.pixels-1,image,[0 255]);
colormap gray; axis image; hold on;
for j=1:numel(detections)
    uv=double(detections(j).centroid_px(:)); plot(uv(1),uv(2),'y+');
end
if ~isempty(selected)
    uv=double(selected.centroid_px(:)); plot(uv(1),uv(2),'ro','MarkerSize',15);
end
title(sprintf('Pixels -> OpenCV; ideal pointing %.1f deg',theta*180/pi));
xlabel('u (zero-based pixels)'); ylabel('v (pixels)');
subplot(2,2,4); yyaxis left;
plot(h(:,1),h(:,9),'k-',h(:,1),h(:,10),'m--'); ylabel('Range (m)');
yyaxis right; plot(h(:,1),hypot(h(:,11),h(:,12)),'b-'); ylabel('Thrust (N)');
xlabel('Physical time (s)'); grid on;
title('Camera update -> EKF -> bounded ZEM -> plant');
end

function make_summary(h,c,out,visible)
f=figure('Visible',visible,'Color','w','Position',[80 80 1100 750]);
subplot(2,2,1); plot(h(:,1),h(:,9),h(:,1),h(:,10),'--'); grid on;
xlabel('Physical time (s)'); ylabel('Range (m)'); legend('Truth','EKF');
subplot(2,2,2); plot(h(:,1),hypot(h(:,2)-h(:,7),h(:,3)-h(:,8))); grid on;
xlabel('Physical time (s)'); ylabel('Position error (m)');
subplot(2,2,3); plot(h(:,1),hypot(h(:,11),h(:,12))); grid on;
yline(c.maxForce,'r--'); xlabel('Physical time (s)'); ylabel('Thrust (N)');
subplot(2,2,4); plot(h(:,1),h(:,13)*180/pi,'.'); grid on;
xlabel('Physical time (s)'); ylabel('Bearing innovation (deg)');
exportgraphics(f,fullfile(out,'estimation_and_control.png'),'Resolution',140);
if ~strcmp(visible,'on'), close(f); end
end

function make_orbit_overview(c,out,visible)
f=figure('Visible',visible,'Color','w','Position',[100 100 800 650]);
times=linspace(0,c.period,200); orbit=zeros(2,numel(times));
for k=1:numel(times), p=binary_state(times(k),c); orbit(:,k)=p(:,2); end
plot(orbit(1,:),orbit(2,:),'r-'); axis equal; hold on; grid on;
th=linspace(0,2*pi,100); plot(c.radius(1)*cos(th),c.radius(1)*sin(th),'k-');
for t=[0,c.period/4,c.period/2,3*c.period/4]
    p=binary_state(t,c); plot(p(1,2),p(2,2),'ro');
    a=2*pi*t/c.spin;
    plot([0,c.radius(1)*cos(a)],[0,c.radius(1)*sin(a)],'Color',[.7 .3 .1]);
end
xlabel('Barycentric x (m)'); ylabel('Barycentric y (m)');
title({'One physical binary revolution: 11 h 55 min (42,900 s)', ...
    'Didymos spin: 2.26 h (8,136 s); sampled overview, separate from terminal run'});
exportgraphics(f,fullfile(out,'full_orbit_overview.png'),'Resolution',140);
if ~strcmp(visible,'on'), close(f); end
end

function a=wrap(a)
a=atan2(sin(a),cos(a));
end

function close_writer(writer)
if ~isempty(writer), close(writer); end
end
