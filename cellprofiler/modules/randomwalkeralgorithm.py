"""

Random walker algorithm

"""

import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import skimage.measure
import skimage.segmentation


class RandomWalkerAlgorithm(cellprofiler.module.ImageSegmentation):
    module_name = "Random walker algorithm"

    variable_revision_number = 1

    def create_settings(self):
        super(RandomWalkerAlgorithm, self).create_settings()

        self.markers_name = cellprofiler.setting.ImageNameSubscriber(
            "Markers"
        )

        self.markers_a = cellprofiler.setting.Float(
            "A",
            0.5
        )

        self.markers_b = cellprofiler.setting.Float(
            "B",
            0.5
        )

        self.beta = cellprofiler.setting.Float(
            "Beta",
            130.0
        )

    def settings(self):
        __settings__ = super(RandomWalkerAlgorithm, self).settings()

        return __settings__ + [
            self.markers_name,
            self.markers_a,
            self.markers_b,
            self.beta
        ]

    def visible_settings(self):
        __settings__ = super(RandomWalkerAlgorithm, self).settings()

        return __settings__ + [
            self.markers_name,
            self.markers_a,
            self.markers_b,
            self.beta
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        markers_name = self.markers_name.value

        markers = images.get_image(markers_name)

        data = x_data

        markers_data = markers.pixel_data

        markers_data = skimage.img_as_float(markers_data)

        labels_data = numpy.zeros_like(markers_data, numpy.uint8)

        labels_data[markers_data > self.markers_a.value] = 1

        labels_data[markers_data < self.markers_b.value] = 2

        y_data = skimage.segmentation.random_walker(
            data=data,
            labels=labels_data,
            beta=self.beta.value,
            mode="cg_mg"
        )

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        workspace.object_set.add_objects(objects, y_name)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
