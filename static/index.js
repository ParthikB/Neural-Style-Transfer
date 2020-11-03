// To show the image //

function readURL(input, tag) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $(tag).attr('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]); // convert to base64 string
  }
}


function onButtonClick(div, div2){
  document.getElementById(div).className="show";
  document.getElementById(div2).className="hide";

}


$("#img1name").change(function () {
  readURL(this, '#img1');
});

$("#img2name").change(function () {
  readURL(this, '#img2');
});