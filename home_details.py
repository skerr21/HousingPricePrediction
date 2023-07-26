from geopy.geocoders import Nominatim

def get_zip_and_state(address):
    geolocator = Nominatim(user_agent="homeDetailsApp")
    location = geolocator.geocode(address)
    if location:
        location = geolocator.reverse([location.latitude, location.longitude], exactly_one=True)
        address = location.raw['address']
        state = address.get('state', '')
        postcode = address.get('postcode', '')
        return state, postcode
    return None, None

address = "1600 Amphitheatre Parkway, Mountain View, CA"  # sample address
state, zipcode = get_zip_and_state(address)
print(f'State: {state}, Zip: {zipcode}')
