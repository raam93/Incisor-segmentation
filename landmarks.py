

"""
	A Class for representing the landmarks of an incisor in an image.
    The landmarks as a list of np.array([float, float]) coordinates.
    Based on: https://github.com/probberechts/IncisorSegmentation/blob/master/Code/landmarks.py
"""


import os
import fnmatch
import numpy as np

DATA_DIR = os.path.join('.', 'Data/Landmarks/original')

class Landmarks(object):
    
    def __init__(self, source):
        """Creates a new set of landmarks.

        Args:
            source : The path of the landmarks file ||
                A list with landmark points in [x_1,...,x_n, y_1,..., y_n] format.

        """
        self.points = np.array([])
        if source is not None:
            # read from file
            if isinstance(source, str):
                self._read_landmarks(source)
            # read from vector
            elif isinstance(source, np.ndarray) and np.atleast_2d(source).shape[0] == 1:
                self.points = np.array((source[:len(source)/2], source[len(source)/2:])).T
            # read from matrix
            elif isinstance(source, np.ndarray) and source.shape[1] == 2:
                self.points = source
            else:
                raise ValueError("Unsupported source type for Landmarks object.")

    def _read_landmarks(self, input_file):
        """Processes the given input_file with landmarks.
           points: A numpy array of landmark points as [[x_1, y_1], ..., [x_n, y_n]].
           (np.array of np.array points)
        """
        lines = open(input_file).readlines()
        points = []
        for x, y in zip(lines[0::2], lines[1::2]):
            points.append(np.array([float(x), float(y)]))
        self.points = np.array(points)

    def as_vector(self):
        """
        Returns the landmark points in [x_1,...,x_n, y_1,..., y_n] numpy array format.
        """
        return np.hstack((self.points[:, 0], self.points[:, 1]))

    def as_matrix(self):
        """
        Returns the lanmark points in [[x_1, y_1], ..., [x_n, y_n]] numpy array format.
        """
        return self.points

    def get_centroid(self):
        """
        Returns the center of mass of this shape.
        """
        return np.mean(self.points, axis=0)

    def get_center(self):
        """Returns the center of this shape.

        Returns:
            The center as [x, y]
        """
        return [self.points[:, 0].min() + (self.points[:, 0].max() - self.points[:, 0].min())/2,
                self.points[:, 1].min() + (self.points[:, 1].max() - self.points[:, 1].min())/2]

    def get_crown(self, is_upper):
        """Returns the top part of the tooth, without the root.

        Returns:
            The crown part of the tooth.
        """
        if is_upper:
            return Landmarks(self.points[10:30, :])
        else:
            points = np.vstack((self.points[0:10, :], self.points[30:40, :]))
            return Landmarks(points)

    def translate_to_origin(self):
        """
        Translates the landmark points so that the centre of gravitiy of this
        shape is at the origin.
        """
        centroid = np.mean(self.points, axis=0)
        points = self.points - centroid
        return Landmarks(points)

    def translate_center_to_origin(self):
        """
        Translates the landmark points so that the center of this
        shape is at the origin.
        """
        center = self.get_center()
        points = self.points - center
        return Landmarks(points)


    def scale_to_unit(self):
        """
        Scales each landmark point so that the norm of the shape is 1.
        """
        centroid = np.mean(self.points, axis=0)
        scale_factor = np.sqrt(np.power(self.points - centroid, 2).sum())
        points = self.points.dot(1. / scale_factor)
        return Landmarks(points)

    def translate(self, vec):
        """Translates this model to given centroid.
        Args:
            vec : (1, 2) numpy array representing the translation vector.
        """
        points = self.points + vec
        return Landmarks(points)

    def scale(self, factor):
        """Rescales this model by the given factor.
        Args:
            factor: The scale factor.
        """
        centroid = self.get_centroid()
        points = (self.points - centroid).dot(factor) + centroid
        return Landmarks(points)

    def scale_to_bbox(self, bbox):
        """Rescales this model to fit in the given bbox.
        Args:
            bbox: The surrounding bbox.
        """
        bbox_h = bbox[1][1] - bbox[0][1]
        # bbox_w = bbox[1][0] - bbox[0][0]
        # scale_w = bbox_w / (self.points[:,0].max() - self.points[:,0].min())
        scale_h = bbox_h / (self.points[:, 1].max() - self.points[:, 1].min())
        return self.scale(scale_h)

    def scaleposition(self, factor):
        """Rescales the coordinates of this model by the given factor.
        (changes scale and position of the landmarks)

        Args:
            factor: The scale factor.
        """
        points = self.points.dot(factor)
        return Landmarks(points)

    def mirror_y(self):
        """
        Mirrors this landmark points around the y axis.
        """
        centroid = self.get_centroid()
        points = self.points - centroid
        points[:, 0] *= -1
        points = points + centroid
        points = points[::-1]
        
        return Landmarks(points)

    def rotate(self, angle):
        """Rotates this model clockwise by the given angle.
        Args:
            angle: The rotation angle (in radians).
        """
        # create rotation matrix
        rotmat = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

        # apply rotation on each landmark point
        points = np.zeros_like(self.points)
        centroid = self.get_centroid()
        tmp_points = self.points - centroid
        for ind in range(len(tmp_points)):
            points[ind, :] = tmp_points[ind, :].dot(rotmat)
        points = points + centroid

        return Landmarks(points)

    def T(self, t, s, theta):
        """Performs a rotation by theta, a scaling by s and a translation
        by t on this model.

        Args:
            t: translation vector.
            s: scaling factor.
            theta: rotation angle (counterclockwise, in radians)
        """
        return self.rotate(theta).scale(s).translate(t)

    def invT(self, t, s, theta):
        """Performs the inverse transformation.

        Args:
            t: translation vector.
            s: scaling factor.
            theta: rotation angle (counterclockwise, in radians)
        """
        return self.translate(-t).scale(1/s).rotate(-theta)



def load_all_incisors_of_example(img_idx):
    """
    Collects all landmarks for a given radiograph.

    """
    files = sorted(fnmatch.filter(os.listdir(DATA_DIR), "landmarks{}-*.txt".format(str(img_idx))))
    lm_objects = []
    for filename in files:
        lm_objects.append(Landmarks("{}/{}".format(DATA_DIR, filename)))
    return lm_objects

def load_landmarks(img_idx, incisor_list):
    """
    Loads the landmarks of a given incisor list for a given radiograph.

    """
    files = []
    for incisor in incisor_list:
        one_file = sorted(fnmatch.filter(os.listdir(DATA_DIR), "landmarks{}-{}.txt".format(str(img_idx), str(incisor))))
        files.extend(one_file)
    
    lm_objects = []
    for filename in files:
        lm_objects.append(Landmarks("{}/{}".format(DATA_DIR, filename)))
    return lm_objects
