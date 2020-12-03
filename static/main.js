$(document).ready(function(){
    h = $('.navbar').height() + 30;
    $('.content').css('margin-top', h);
    glist = $('.genre-list');
    for (i = 0; i < glist.length; i++) {
        d = glist.eq(i);
        d.slick({infinite: true,
                 slidesToShow: 10,
                 slidesToScroll: 3,
                 dots: true,
                 arrows: true,
                 variableWidth: true
                });
    }
});

function cartAdd(x) {
    button = $(x);
    button.prop('disabled', true);
    id = button.parent().data('id');
    $.ajax({'url': '/add_to_cart',
            'method': 'post',
            'data': {'ident': id},
            'success': addedToCart,
            'error': cartAddFailed});
}

function addedToCart() {
    $('button').text('Added to cart.');
    cn = $('span#cart-numb').text();
    $('span#cart-numb').text(Number(cn) + 1);
}

function cartAddFailed() {
    $('button').text('Failed to add to cart.');
}