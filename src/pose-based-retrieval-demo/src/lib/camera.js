/*##################################################################
# Methods for handling the webcam to use a self-take picture as query
#
# @author: Angel Villar-Corrales
####################################################################*/

// object processing all the camera and emerging window actions
const camera = function(){

  var imgURI = undefined
  var externalWindow = undefined
  var video = undefined
  var canvas = undefined
  var context = undefined
  var send_uri = false

  // initializing a pop-up window where video is streamed to take a snapshot
  const initialize_window = function(){

    // creating popup window and area to display video
    const w = window.screen.availWidth * 60 / 100;
    const h = window.screen.availHeight * 60 / 100;
    const hv = window.screen.availHeight * 50 / 100;
    externalWindow = window.open('', '', "width="+w+",height="+h+"");
    const video = externalWindow.document.createElement("video");
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = hv;
    canvas.id = 'canvas';
    video.id = "video";
    video.width = w
    video.height = hv
    video.autoplay = true
    video.muted="muted"

    // creating the button to take the snapshot
    var div = document.createElement("DIV");
    div.id = "but_div"
    div.style.width = "100%";
    var btn = document.createElement("BUTTON");
    btn.style.height = "3em";
    btn.style.width = "3em";
    btn.style.marginLeft = "45%";
    btn.style.color = "red";
    btn.style.backgroundColor = "red";
  	btn.style.borderRadius = "50px";
    btn.addEventListener("click", take_snapshot);

    // inserting elements into popup widow
    externalWindow.document.body.append(video)
    externalWindow.document.body.appendChild(div);
    externalWindow.document.getElementById("but_div").appendChild(btn);
    externalWindow.document.body.appendChild(canvas);
    externalWindow.addEventListener('unload', function(eventObject) {
      console.log("unload")
      send_uri = true
    });
    externalWindow.focus();

    return externalWindow
  }

  // streaming video to a canvas
  const stream_video = function (video) {
    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
      video.srcObject = stream;
      video.play();
    });
  }

  const take_snapshot = function(){
    console.log("Click")
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    imgURI = canvas.toDataURL('image/png');
    externalWindow.close()
  }

  // exporting methods to be called from the take_picture component
  return{

    start_camera: function(){
      let externalWindow = initialize_window()
      externalWindow.focus();
      video = externalWindow.document.getElementById("video")
      canvas = externalWindow.document.getElementById("canvas")
      context = canvas.getContext("2d")
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        stream_video(video)
      }
    },
    get_img_uri: function(){
      
      return imgURI
    }

  }



}();

export default camera;

//
