
class Finder(object):
      
    def execute(self, obj, boxesonly=False):
    	if boxesonly:
           return self.find_boxes(obj)
    	else:
	   return self.find_objects(obj)

    def find_boxes(self, obj):
        raise NotImplementedError

    def find_objects(self, obj):
        raise NotImplementedError

    def get_label(self, index):
        assert 0 <= index < len(self.labels)
        return self.labels[index]

    def get_label_index(self, label):
        assert label in self.labels
        return self.labels.index(label)


 
