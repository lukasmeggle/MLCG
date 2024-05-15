from PyRT_Common import *
from random import randint, choice

from PyRT_Core import *
from GaussianProcess import *


# -------------------------------------------------
# Integrator Classes
# -------------------------------------------------
# The integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
# -------------------------------------------------
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object

        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                d = cam.get_direction(x, y)
                ray = Ray(direction=d)
                pixel = self.compute_color(ray)
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Lazy')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RED
        else:
            return BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=5):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            color = max(1-hit.hit_distance/self.max_depth, 0)
            color = RGBColor(color, color, color)
            return color
        return BLACK
        

class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            color = (hit.normal + Vector3D(1,1,1))/2
            color = RGBColor(color.x, color.y, color.z)
            return color
        return BLACK


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            La = self.scene.i_a
            Ld = BLACK
            
            primitive = self.scene.object_list[hit.primitive_index]
            La = La.multiply(primitive.get_BRDF().kd)
            for light in self.scene.pointLights:
                L = light.pos - hit.hit_point
                L_normalized = Normalize(L)
                L_norm = L.norm()
                ray_light = Ray(hit.hit_point, L_normalized, L_norm)
                hit_light = self.scene.closest_hit(ray_light)
                # if the light is blocked by another object
                if hit_light.has_hit:
                    continue
                # if the light is not blocked compute the diffuse component
                Ld_temp =  primitive.get_BRDF().get_value(L_normalized, 0, hit.normal) 
                Ld = Ld + Ld_temp.multiply(light.intensity) / L_norm **2
               
            return La + Ld
            
        return BLACK


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, filename_, n=10, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n
        self.pdf = UniformPDF()

    def compute_color(self, ray):
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            Lr = BLACK
            samples_dir, samples_prob = sample_set_hemisphere(self.n_samples, self.pdf) 
            primitive = self.scene.object_list[hit.primitive_index]
            for dir, prob in zip(samples_dir, samples_prob):
                # Center the sample around the surface normal
                centered_dir = center_around_normal(dir, hit.normal)
                ray_j = Ray(hit.hit_point, centered_dir)
                hit_j = self.scene.closest_hit(ray_j)
                
                brdf = primitive.get_BRDF().get_value(wi=ray_j.d, wo=ray.d, normal=hit.normal)
                cos_theta = Cosine(hit.normal, ray_j.d)   
                
                if hit_j.has_hit:
                    primitive_j = self.scene.object_list[hit_j.primitive_index]
                    Li = primitive_j.emission
                elif self.scene.env_map is not None:
                    Li = self.scene.env_map.getValue(ray_j.d)
                else:
                    Li = WHITE # neutral element for multiplication
                Lr += Li.multiply(brdf) * cos_theta / prob
            Lr = Lr / self.n_samples
            
        elif self.scene.env_map is not None:
            Lr = self.scene.env_map.getValue(ray.d)
                        
        return Lr


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, filename_, n, num_gp=3, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_GP_' + num_gp + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        
        self.num_gp = num_gp
        self.gp_list = [GaussianProcess(SobolevCov(), Constant(1), noise_=0.01) for i in range(self.num_gp)]
        # Initialize the GP with n samples
        for gp in self.gp_list:
            gp.initialize(self.n_samples)

    def compute_color(self, ray):
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            primitive = self.scene.object_list[hit.primitive_index]

            # Sample one GP
            gp = choice(self.gp_list)
            samples_dir = gp.samples_pos
            # Rotate the hemisphere
            random_rot = randint(0, 360)
            rotated_samples_dir = [rotate_around_y(random_rot, sample) for sample in samples_dir]
            centered_dir = [center_around_normal(sample, hit.normal) for sample in rotated_samples_dir]


            integrands_samples = []
            for dir in centered_dir:
                ray_j = Ray(hit.hit_point, dir)
                hit_j = self.scene.closest_hit(ray_j)

                brdf = primitive.get_BRDF().get_value(wi=ray_j.d, wo=ray.d, normal=hit.normal)
                cos_theta = Cosine(hit.normal, ray_j.d)

                if hit_j.has_hit:
                    primitive_j = self.scene.object_list[hit_j.primitive_index]
                    Li = primitive_j.emission
                elif self.scene.env_map is not None:
                    Li = self.scene.env_map.getValue(ray_j.d)
                else:
                    Li = WHITE
                    
                Lr_i = Li.multiply(brdf) * cos_theta
                integrands_samples.append(Lr_i)
            
            # Compute the estimate
            gp.add_sample_val(integrands_samples)
            Lr = gp.compute_integral_BMC()
            

        elif self.scene.env_map is not None:
            Lr = self.scene.env_map.getValue(ray.d)
        return Lr
