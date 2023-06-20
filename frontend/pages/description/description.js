// pages/description/description.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    imgurl: "",
    name: '',
    proba: '',
    description:''
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {
    let res_imgurl = wx.getStorageSync("res_imgurl");
    // 拿到后台返回的结果
    let res_data = wx.getStorageSync("res_data")["result"];
    // 接收url中传来的key
    let key = options.key;
    let proba = res_data[key]['proba']
    let name =  res_data[key]['name']
    // 更新data中的数据
    this.setData({
      imgurl: res_imgurl,
      name: name,
      proba: proba,
      description: res_data[key]['description']
    });
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})