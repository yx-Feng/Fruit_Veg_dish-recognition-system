Page({
  data: {
    imgurl: "",
    itemData: [],
    result_num:0
  },
  
  onLoad: function() {
      let res_imgurl = wx.getStorageSync("res_imgurl");
      // 拿到后台返回的结果
      let res_data = wx.getStorageSync("res_data");
      let itemData = res_data["result"];
      // 更新data中的数据
      this.setData({
        imgurl: res_imgurl,
        itemData: itemData,
        result_num: Object.keys(itemData).length
      });
  }
});