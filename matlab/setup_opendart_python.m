function info = setup_opendart_python(pythonExecutable, requireOpenCV5)
%SETUP_OPENDART_PYTHON Configure MATLAB -> Python -> OpenCV pixel processing.
% Use a CPython release supported by your MATLAB release. R2025a supports
% Python 3.9--3.12. The environment must contain NumPy and OpenCV 5.
% The optional false flag explicitly enables OpenCV 4 compatibility testing.
if nargin < 1, pythonExecutable = ''; end
if nargin < 2, requireOpenCV5 = true; end
if strlength(string(pythonExecutable)) == 0
    pythonExecutable = getenv('OPENDART_PYTHON');
end
pe = pyenv;
if strlength(string(pythonExecutable)) > 0
    if string(pe.Status) == "Loaded" && ...
            (string(pe.Executable) ~= string(pythonExecutable) || ...
             string(pe.ExecutionMode) ~= "OutOfProcess")
        error('OpenDART:PythonAlreadyLoaded', ...
            ['Python is already loaded. If OutOfProcess, run terminate(pyenv); ' ...
             'otherwise restart MATLAB. Then run setup_opendart_python again.']);
    end
    if string(pe.Status) ~= "Loaded"
        pyenv('Version', char(pythonExecutable), 'ExecutionMode', 'OutOfProcess');
    end
elseif string(pe.Status) ~= "Loaded"
    pyenv('ExecutionMode', 'OutOfProcess');
elseif string(pe.ExecutionMode) ~= "OutOfProcess"
    warning('OpenDART:InProcess', ...
        'Python is already InProcess. Restart MATLAB to select OutOfProcess.');
end
packageRoot = fileparts(fileparts(mfilename('fullpath')));
if int64(py.sys.path.count(packageRoot)) == 0
    insert(py.sys.path, int32(0), packageRoot);
end
bridge = py.importlib.import_module('opendart.matlab_bridge');
info = jsondecode(char(bridge.runtime_info_json(logical(requireOpenCV5))));
fprintf('MATLAB calls Python %s, NumPy %s, OpenCV %s\n', ...
    info.python_version, info.numpy_version, info.opencv_version);
end

