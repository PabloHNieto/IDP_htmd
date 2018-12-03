#!/usr/bin/env python


class Render():
    import os
    from jinja2 import Environment, FileSystemLoader

    PATH = os.path.dirname(os.path.abspath(__file__))
    TEMPLATE_ENVIRONMENT = Environment(
        autoescape=False,
        loader=FileSystemLoader(os.path.join(PATH, 'templates')),
        trim_blocks=False)

    def __init__(self, template, outName, info):
        self.outName = outName + '.html'
        self.template = template + '.html'
        self.info = info
        self.create_index_html()

    def render_template(self, context):
        return self.TEMPLATE_ENVIRONMENT.get_template(
            self.template).render(context)

    def create_index_html(self):
        with open(self.outName, 'w') as f:
            html = self.render_template(self.info)
            f.write(html)

########################################


if __name__ == "__main__":
    urls = ['http://example.com/1',
            'http://example.com/2', 'http://example.com/3']
    
    context = {
        'urls': urls
    }

    r = Render("test_template", "PHN_out", context)
