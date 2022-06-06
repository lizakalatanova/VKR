if self.drop_down_1.selected_value=="Процент сжатия":
  lrn=int(self.text_box_1.text)/100
else:
  lrn=int(self.text_box_1.text)
self.text_area_3.text = anvil.server.call('predict_iris',self.text_area_1.text)
self.text_area_4.text = anvil.server.call('predict_title',self.text_area_3.text).capitalize()
self.text_area_5.text = anvil.server.call('predict_key',self.text_area_1.text, int(self.text_box_2.text)).capitalize()
self.text_area_2.text = anvil.server.call('predict_sum',self.text_area_1.text,lrn )
