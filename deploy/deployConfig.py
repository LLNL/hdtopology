import os
import sys
import subprocess
import glob
import shutil
import platform
import errno
import fnmatch
import os
import sysconfig
import re

bVerbose=False

WIN32=platform.system()=="Windows" or platform.system()=="win32"
APPLE=platform.system()=="Darwin"
LINUX=not APPLE and not WIN32

# /////////////////////////////////////////////////
class DeployUtils:

    # ExecuteCommand
    @staticmethod
    def ExecuteCommand(cmd):
        """
        note: shell=False does not support wildcard but better to use this version
        because quoting the argument is not easy
        """
        print("# Executing command: ",cmd)
        return subprocess.call(cmd, shell=False)

    @staticmethod
    def ExecuteCommandReturnOutput(cmd):
        print("# Executing command: ", cmd)

        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout,stderr = out.communicate()
        return stdout, stderr

    # GetCommandOutput
    @staticmethod
    def GetCommandOutput(cmd):
        output=subprocess.check_output(cmd)
        if sys.version_info >= (3, 0): output=output.decode("utf-8")
        return output.strip()

    # CreateDirectory
    @staticmethod
    def CreateDirectory(value):
        try:
            os.makedirs(value)
        except OSError:
            if not os.path.isdir(value):
                raise

    # GetFilenameWithoutExtension
    @staticmethod
    def GetFilenameWithoutExtension(filename):
        return os.path.splitext(os.path.basename(filename))[0]

    # CopyFile
    @staticmethod
    def CopyFile(src,dst):

        src=os.path.realpath(src)
        dst=os.path.realpath(dst)

        if src==dst or not os.path.isfile(src):
            return

        DeployUtils.CreateDirectory(os.path.dirname(dst))
        shutil.copyfile(src, dst)

    # CopyDirectory
    @staticmethod
    def CopyDirectory(src,dst):

        src=os.path.realpath(src)

        if not os.path.isdir(src):
            return

        DeployUtils.CreateDirectory(dst)

        # problems with symbolic links so using shutil
        dst=dst+"/" + os.path.basename(src)

        if os.path.isdir(dst):
            shutil.rmtree(dst,ignore_errors=True)

        shutil.copytree(src, dst, symlinks=True)

    # ReadTextFile
    @staticmethod
    def ReadTextFile(filename):
        file = open(filename, "r")
        ret=file.read().strip()
        file.close()
        return ret

    # WriteTextFile
    @staticmethod
    def WriteTextFile(filename,content):
        if not isinstance(content, str):
            content="\n".join(content)+"\n"
        DeployUtils.CreateDirectory(os.path.dirname(os.path.realpath(filename)))
        file = open(filename,"wt")
        file.write(content)
        file.close()

    # ExtractNamedArgument
    #example arg:=--key='value...'
    # ExtractNamedArgument("--key")
    @staticmethod
    def ExtractNamedArgument(key):
        for arg in sys.argv:
            if arg.startswith(key + "="):
                ret=arg.split("=",1)[1]
                if ret.startswith('"') or ret.startswith("'"): ret=ret[1:]
                if ret.endswith('"')   or ret.endswith("'"):   ret=ret[:-1]
                return ret

        return ""

    # RemoveFiles
    @staticmethod
    def RemoveFiles(pattern):
        files=glob.glob(pattern)
        print("Removing files",files)
        for it in files:
            if os.path.isfile(it):
                os.remove(it)
            else:
                shutil.rmtree(os.path.abspath(it),ignore_errors=True)

    # RecursiveFindFiles
    # glob(,recursive=True) is not supported in python 2.x
    # see https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    @staticmethod
    def RecursiveFindFiles(rootdir='.', pattern='*'):
      return [os.path.join(looproot, filename)
              for looproot, _, filenames in os.walk(rootdir)
              for filename in filenames
              if fnmatch.fnmatch(filename, pattern)]

    # PipInstall
    @staticmethod
    def PipInstall(packagename,extra_args=[]):
        cmd=[sys.executable,"-m","pip","install","--user",packagename]
        if extra_args: cmd+=extra_args
        print("# Executing",cmd)
        return_code=subprocess.call(cmd)
        return return_code==0



class RpathDeploy:

    """
     reconfigure the rpath to make the dynamic library more portable
    """

    # constructor
    def __init__(self):
        pass


    # __findAllBinaries
    def __findAllBinaries(self):
        ret=[]

        ret+=DeployUtils.RecursiveFindFiles('.', '*.so')
        ret+=DeployUtils.RecursiveFindFiles('.', '*.dylib')

        # apps
        for it in glob.glob("bin/*.app"):
            bin="%s/Contents/MacOS/%s" % (it,DeployUtils.GetFilenameWithoutExtension(it))
            if os.path.isfile(bin):
                ret+=[bin]

        # frameworks
        for it in glob.glob("bin/*.framework"):
            file="%s/Versions/Current/%s" % (it,DeployUtils.GetFilenameWithoutExtension(it))
            if os.path.isfile(os.path.realpath(file)):
                ret+=[file]

        return ret

    # __extractDeps
    def __extractDeps(self,filename):
        output=DeployUtils.GetCommandOutput(['otool', '-L' , filename])
        lines=output.split('\n')[1:]
        deps=[line.strip().split(' ', 1)[0].strip() for line in lines]

        # remove any reference to myself
        deps=[dep for dep in deps if os.path.basename(filename)!=os.path.basename(dep)]
        return deps

    # __findLocal
    def __findLocal(self,filename):
        key=os.path.basename(filename)
        return self.locals[key] if key in self.locals else None

    # __addLocal
    def __addLocal(self,filename):

        # already added
        if self.__findLocal(filename):
            return

        key=os.path.basename(filename)

        print("# Adding local",key,"=",filename)

        self.locals[key]=filename

        for dep in self.__extractDeps(filename):
            self.__addGlobal(dep)


    # __addGlobal
    def __addGlobal(self,dep):

        # it's already a local
        if self.__findLocal(dep):
            return

        # wrong file
        if not os.path.isfile(dep) or not dep.startswith("/"):
            return

        # ignoring the OS system libraries
        if dep.startswith("/System") or dep.startswith("/lib") or dep.startswith("/usr/lib"):
            return

        # I don't want to copy Python dependency
        if "Python.framework" in dep :
            print("Ignoring Python.framework:",dep)
            return

        key=os.path.basename(dep)

        print("# Adding global",dep,"=",dep)

        # special case for frameworks (I need to copy the entire directory)
        if APPLE and ".framework" in dep:
            framework_dir=dep.split(".framework")[0]+".framework"
            DeployUtils.CopyDirectory(framework_dir,"bin")
            filename="bin/" + os.path.basename(framework_dir) + dep.split(".framework")[1]
            self.__addLocal(filename) # now a global becomes a local one
            return

        if dep.endswith(".dylib") or dep.endswith(".so"):
            filename="bin/" + os.path.basename(dep)
            DeployUtils.CopyFile(dep,filename)
            self.__addLocal(filename) # now a global becomes a local one
            return

        raise Exception("Unknonw dependency %s file file" % (dep,))


    def customizedDeploy(self):
        print("All dynamic libraries: ", self.__findAllBinaries())
        for filename in self.__findAllBinaries():
            if APPLE:
                # try:
                #     #FIXME temp fix when path path is used instead rpath in the dynamic library
                #     output, _ = DeployUtils.ExecuteCommandReturnOutput(['otool', '-l', filename])
                #     outputList = str(output).split("\\n")
                #     pythonPath = [x for x in outputList if "python" in x]
                #     print("pythonPath", pythonPath)
                #     pythonPathName = pythonPath[0].split()[1]
                #     print("pythonPathName", pythonPathName)
                #     DeployUtils.ExecuteCommand(['install_name_tool','-change', pythonPathName ,"@rpath/"+ "libpython3.7m.dylib",filename])
                # except Exception as e:
                #     print(e)
                #     print("python dylib is already linked as a rpath!")

                print(filename)
                output, _ = DeployUtils.ExecuteCommandReturnOutput(['otool', '-l', filename])
                outputList = str(output).split("\\n")
                # print("outputList", outputList)
                pythonPath = [x for x in outputList if "Python.framework" in x]
                print("pythonPath", pythonPath)
                pythonPathName = pythonPath[0].split()[1]
                print("pythonPathName", pythonPathName)
                DeployUtils.ExecuteCommand(['install_name_tool','-change', pythonPathName ,"@rpath/"+ "libpython3.7m.dylib",filename])

                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../lib',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../../lib',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../../../',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../../../../Cellar/python/3.7.4/Frameworks/Python.framework/Versions/3.7/lib',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../../../../Cellar/python/3.7.4_1/Frameworks/Python.framework/Versions/3.7/lib',filename])
                DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path/../../../../Cellar/python/3.7.4_2/Frameworks/Python.framework/Versions/3.7/lib',filename])
            elif LINUX:
                listPath = ['$ORIGIN/']
                listPath.append('/usr/lib/x86_64-linux-gnu/')
                DeployUtils.ExecuteCommand(['patchelf','--set-rpath', ":".join(v),filename])


    def copyExternalDependenciesAndFixRPaths(self):

        self.locals={}

        for filename in self.__findAllBinaries():
            self.__addLocal(filename)

        print(self.locals)

        # note: __findAllBinaries need to be re-executed
        for filename in self.__findAllBinaries():

            deps=self.__extractDeps(filename)

            if bVerbose:
                print("")
                print("#/////////////////////////////")
                print("# Fixing",filename,"which has the following dependencies:")
                for dep in deps:
                    print("#\t",dep)
                print("")

            def getRPathBaseName(filename):
                if ".framework" in filename:
                    # example: /bla/bla/ QtOpenGL.framework/Versions/5/QtOpenGL -> QtOpenGL.framework/Versions/5/QtOpenGL
                    return os.path.basename(filename.split(".framework")[0]+".framework") + filename.split(".framework")[1]
                else:
                    return os.path.basename(filename)

            DeployUtils.ExecuteCommand(["chmod","u+rwx",filename])
            DeployUtils.ExecuteCommand(['install_name_tool','-id','@rpath/' + getRPathBaseName(filename),filename])

            # example QtOpenGL.framework/Versions/5/QtOpenGL
            for dep in deps:
                local=self.__findLocal(dep)
                if local:
                    DeployUtils.ExecuteCommand(['install_name_tool','-change',dep,"@rpath/"+ getRPathBaseName(local),filename])

            DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path',filename])

            # how to go from a executable in ./** to ./bin
            root=os.path.realpath("")
            curr=os.path.dirname(os.path.realpath(filename))
            N=len(curr.split("/"))-len(root.split("/"))
            DeployUtils.ExecuteCommand(['install_name_tool','-add_rpath','@loader_path' +  ("/.." * N) + "/bin",filename])


    # addRPath
    def addRPath(self,value):
        for filename in self.__findAllBinaries():
            DeployUtils.ExecuteCommand(["install_name_tool","-add_rpath",value,filename])

RpathDeploy().customizedDeploy()
