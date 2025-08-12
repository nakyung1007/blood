# blood.py
from mmdet.datasets import DATASETS, CocoDataset

@DATASETS.register_module()
class BloodsDataset(CocoDataset):
    CLASSES = ('neutrophil',)  # 카테고리 이름을 실제로 맞춰주세요

    def load_annotations(self, ann_file):
        # CocoDataset.load_annotations를 그대로 쓰거나,
        # raw JSON 포맷이 다르면 여기서 파싱 로직을 오버라이드
        return super().load_annotations(ann_file)
