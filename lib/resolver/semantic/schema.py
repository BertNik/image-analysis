

LABEL_ENTITIES = {
    "brand_name": (
        "The commercial product brand name exactly as displayed "
        "on the label. Exclude alcoholic beverage class or type "
        "designations such as Cognac, Bourbon Whiskey, Vodka, "
        "Gin, Rum, Tequila, Brandy, Wine, Beer, Ale, and Lager. "
        "For example, when 'MAISON BELLE' is followed by "
        "'COGNAC', the brand name is 'MAISON BELLE' only."
    ),
    "class_type": (
        "The complete regulated alcoholic beverage class or type "
        "exactly as displayed on the label. The designation may "
        "be one word, such as 'COGNAC', 'VODKA', 'GIN', or 'RUM', "
        "or multiple adjacent lines, such as 'KENTUCKY STRAIGHT' "
        "followed by 'BOURBON WHISKEY'. Exclude brand names and "
        "marketing terms."
    ),
    "business_name": (
        "The name of a business or company identified on the product label."
        "Complete name of a business or company identified on "
        "the product label, such as a producer, distiller, "
        "bottler, or importer. Extract the company name without "
        "assigning a business role from words contained inside "
        "the company name itself."
    ),
    "business_address": (
        "The complete mailing or physical address printed for "
        "a business identified on the product label. Include "
        "street, city, state, postal code, and country when "
        "present. A city, state, and postal code alone may be "
        "a complete label address. Examples include "
        "'LOUISVILLE, KENTUCKY 40202' and "
        "'123 Bourbon Lane, Frankfort, KY 40601 USA'."
    ),
    "country_of_origin": (
        "The country where the alcoholic beverage product "
        "originates. Extract the country itself, regardless "
        "of the language used on the label. Do not extract "
        "the importer location or another business address."
    ),

    "country_of_origin_statement": (
        "The complete label statement that identifies the "
        "alcoholic beverage product's country of origin, "
        "regardless of the language used. Extract the full "
        "statement containing the origin country. Do not "
        "extract a producer location, importer location, "
        "or general business address."
    ),
}



BUSINESS_RELATIONS = {
    "distilled_by": (
        "The target is the business or company responsible for distilling "
        "the product, regardless of the language used in the text."
    ),
    "bottled_by": (
        "The target is the business or company responsible for bottling "
        "the product, regardless of the language used in the text."
    ),
    "produced_by": (
        "The target is the business or company responsible for producing "
        "the product, regardless of the language used in the text."
    ),
    "imported_by": (
        "The target is the business explicitly identified on "
        "the label as importing this alcoholic beverage into "
        "the United States. Create this relation only when the "
        "label text explicitly establishes an importer "
        "relationship, such as 'IMPORTED BY' or equivalent "
        "wording. Do not infer this relation merely because a "
        "business name contains words such as 'Import', "
        "'Imports', 'Importer', or 'International'."
    ),
    "business_located_at": (
        "The source is the complete business or company name "
        "and the target is the complete mailing or physical "
        "address belonging to that business. Create the relation "
        "when the address appears directly beneath, beside, or "
        "as part of the same producer, distiller, bottler, or "
        "importer statement. For example, relate "
        "'HARBOR STONE DISTILLING COMPANY' to "
        "'LOUISVILLE, KENTUCKY 40202'."
    ),
}


BUSINESS_ROLE_BY_RELATION = {
    "distilled_by": "distiller",
    "bottled_by": "bottler",
    "produced_by": "producer",
    "imported_by": "importer",
}

