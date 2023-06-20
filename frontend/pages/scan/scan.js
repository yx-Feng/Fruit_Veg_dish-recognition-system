Page({
  data: {},
  Recognition: function(n) {
    var o = n.currentTarget.id; // "camera"还是"album"
    let that = this;
    wx.chooseMedia({
      count: 1,
      sizeType: [ "compressed" ], // 是否压缩所选文件
      sourceType: [ o ],  // 图片选择的来源
      success: function (res) {
        let img_file = res.tempFiles[0].tempFilePath;
        wx.setStorageSync("res_imgurl", img_file); // 将数据存储在本地缓存中指定的key中
        //console.log("上传时："+img_file) //最终图片路径
        wx.showLoading({
          title: "正在识别中..."
        })
        wx.uploadFile({
          url: 'https://www.designtuesday.top:453/predict', //仅为示例，非真实的接口地址
          filePath: img_file,
          name: 'image',
          success (res){
            let res_data = JSON.parse(res.data);
            wx.setStorageSync("res_data", res_data);
            console.log(res_data)
            wx.navigateTo({
              url: "/pages/info/info"
            });
          },
          fail: function (res) {
            wx.showModal({
              title: "提示",
              content: "获取出错，请重试",
              showCancel: false,
              success: function (res) {
                if (res.confirm) {
                } else if (res.cancel) {
                }
              }
            })
          },
          complete: function (res) {
            wx.hideLoading();  // 关闭加载框
          }
        })
      }
    });
  }
});
