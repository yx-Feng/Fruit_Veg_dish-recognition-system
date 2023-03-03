//========================================================================
// 拖放图像处理
//========================================================================

let fileDrag = document.getElementById("file-drag");
let fileSelect = document.getElementById("file-upload");
// 添加事件句柄
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();
  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  let files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (let i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// 要调用函数的网页元素
//========================================================================

let imageDisplay = document.getElementById("image-display");
let uploadCaption = document.getElementById("upload-caption");
let predResult = document.getElementById("pred-result");
let loader = document.getElementById("loader");

//========================================================================
// 主要按钮事件
//========================================================================

function submitImage() {
  // 提交图片
  // 没选择一张图片就点提交按钮了
  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }
  // 显示loader加载器动图
  loader.classList.remove("hidden");
  // 调用后端的预测函数
  predictImage(imageDisplay.src);
}

function clearImage() {
  // 重置选择的图片
  fileSelect.value = "";
  // 清除<img>标签的src属性，清除预测的结果
  imageDisplay.src = "";
  predResult.innerHTML = "";

  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);  // 显示提示
}

function previewFile(file) {
  // 展示预览图片
  let fileName = encodeURI(file.name);
  let reader = new FileReader();
  reader.readAsDataURL(file);  // 读取指定的Blob或File对象
  reader.onloadend = () => {
    imageDisplay.src = reader.result;  // result属性包含一个data:URL格式的字符串(base64编码）以表示所读取文件的内容
    show(imageDisplay);   // 展示图片
    hide(uploadCaption);  // 隐藏提示
    // 重置
    predResult.innerHTML = "";
  };
}

//========================================================================
// 辅助函数
//========================================================================

function predictImage(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  }).then(resp => {
    if (resp.ok)
      resp.json().then(data => {
        displayResult(data);
      });
  }).catch(err => {
    console.log("An error occurred", err.message);
    window.alert("Oops! Something went wrong.");
  });
}

function displayResult(data) {
  // 展示结果
  hide(loader);
  // 传过来data出现了乱序, 按属性值降序排一下
  const keys = Object.keys(data).sort(function(a, b) {
    return parseFloat(data[b]) - parseFloat(data[a])
  })
  for (let i = 0; i < keys.length; i++) {
    let elem = document.createElement("div");
    elem.appendChild(document.createTextNode(keys[i]+": "+data[keys[i]]+ "%"))
    predResult.appendChild(elem);
  }
  show(predResult);
}

function hide(el) {
  // 隐藏一个元素
  el.classList.add("hidden");
}

function show(el) {
  // 展示一个元素
  el.classList.remove("hidden");
}