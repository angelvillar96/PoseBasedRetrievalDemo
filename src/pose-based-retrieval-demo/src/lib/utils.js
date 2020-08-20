/*##################################################################
# Utils methods for auxiliary purposes, e.g., encoding, input/output
# and displaying
#
# @author: Angel Villar-Corrales
####################################################################*/


// converting a base64 object received by POST into a url object to be fed into
// the an image canvas or some other display
export const decodeBase64 = (encoded) =>{
    var raw = window.atob(encoded);
    var rawLength = raw.length;
    var binary_rep = new Uint8Array(new ArrayBuffer(rawLength));

    // converting each encoded char to binary
    for(var i = 0; i < rawLength; i++) {
      binary_rep[i] = raw.charCodeAt(i);
    }
    // converting into image blob and to URL object
    var blob = new Blob([binary_rep], {type: 'blob'});

    return blob
}


// converting the URI from an image taken using the camera to a File Object
export const objectUrl2File = (dataURI) =>{

  var byteString = atob(dataURI.split(',')[1]);
  var ab = new ArrayBuffer(byteString.length);
  var ia = new Uint8Array(ab);
  for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
  }
  var blob = new Blob([ia], { type: 'image/jpeg' });
  var file = new File([blob], "image.jpg");

  return file
}


// obtaining current time as string YYYY-MM-DD-HH-MM-SS
export const get_timestamp = () =>{
  function pad2(n) {  // always returns a string
            return (n < 10 ? '0' : '') + n;
        }
  var date = new Date();
  return date.getFullYear() + "-" +
         pad2(date.getMonth() + 1) + "-" +
         pad2(date.getDate()) + "-" +
         pad2(date.getHours()) + "-" +
         pad2(date.getMinutes()) + "-" +
         pad2(date.getSeconds());
}


// method to test correct import
export const lib_is_on = () =>{
  alert("Library is imported")
}


//
