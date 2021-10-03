$(document).ready(function () {

  $(".send_btn").click(function () {
    enviarDatos($("#message").val());
  });

  $("#message").keypress(function (event) {
    if (event.keyCode === 13) {
      enviarDatos($("#message").val());
    }
  });

});

function enviarDatos(input_text) {
  if (!input_text) {
    return;
  }

  let date = new Date().toLocaleTimeString();

  $(".msg_card_body").append(`
	  <div class="d-flex justify-content-end mb-4"> <div class="msg_cotainer_send">${input_text}<span class="msg_time_send">${date}</span> </div> <div class="img_cont_msg"> <img src="static/img/user.jpg" class="rounded-circle user_img_msg"> </div> </div>
	`);

  $("#message").val("");

  var d = $(".msg_card_body");
  d.scrollTop(d.prop("scrollHeight"));

  // llamar al servicio de post

  $(".user_info p").show();
  // var ajaxTime= new Date().getTime();
  $.ajax({
    // type: "GET",
    // url: "https://httpbin.org/delay/5",
    type: "POST",
    url: "/send",
    data: { message: input_text},
  }).done(function (response) {
    // var totalTime = new Date().getTime()-ajaxTime;
    $(".user_info p").hide();

    //alert(response.message)

    let output_text = response.message;

    $(".msg_card_body").append(`
		<div class="d-flex justify-content-start mb-4"> <div class="img_cont_msg"> <img src="static/img/chatbot.png" class="rounded-circle user_img_msg"> </div> <div class="msg_cotainer">${output_text}<span class="msg_time">${date}</span> </div> </div>
		  `);

    var d = $(".msg_card_body");
    d.scrollTop(d.prop("scrollHeight"));
  });
}
